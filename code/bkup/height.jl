# File: Observables/Height.jl
# Contains all observables related to the Height operator (h).

using LinearAlgebra
using ..Observables

#--------------------------------------------------------------------
# 1. Height Expectation Value
#--------------------------------------------------------------------
mutable struct HeightExpect <: AbstractObservable
    mean_sum::Vector{Float64}
    final_value::Vector{Float64}
    HeightExpect(L::Int) = new(zeros(Float64, L), zeros(Float64, L))
end

get_filename(obs::HeightExpect) = "h_expect.dat"
get_header(obs::HeightExpect) = "Site\tH_Avg\tH_Err"
get_data(obs::HeightExpect) = obs.final_value
get_xaxis(obs::HeightExpect, L::Int, tmea::Vector{Int}) = 1:L

function measure!(obs::HeightExpect, sys, workspace)
    # H_vals (the height profile) is already calculated in the workspace
    obs.mean_sum .+= workspace.H_vals
end

function finalize!(obs::HeightExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value .= obs.mean_sum ./ total_measurements
    end
end

#--------------------------------------------------------------------
# 2. Height-Height Time Correlation at the center point L/2
#--------------------------------------------------------------------
mutable struct HeightTimeCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    H_mid_history::Vector{Float64} # Local history for the midpoint height
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    
    function HeightTimeCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, buffer_size, zeros(Float64, buffer_size),
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::HeightTimeCorr) = "h_tcorr.dat"
get_header(obs::HeightTimeCorr) = "t\tHH_t_Avg\tHH_t_Err"
get_data(obs::HeightTimeCorr) = obs.final_corr
get_xaxis(obs::HeightTimeCorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::HeightTimeCorr, sys, workspace)
    current_H_mid = workspace.H_vals[sys.L ÷ 2]
    
    buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
    obs.H_mid_history[buffer_idx] = current_H_mid

    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_idx = (workspace.t - lag - 1) % obs.buffer_size + 1
            past_H_mid = obs.H_mid_history[past_idx]
            
            obs.prod_sum[idx] += current_H_mid * past_H_mid
            obs.mean_t_sum[idx] += current_H_mid
            obs.mean_0_sum[idx] += past_H_mid
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::HeightTimeCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# 3. Height Diffusion A(t) 
#--------------------------------------------------------------------
mutable struct HeightDiffusion <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    H_mid_history::Vector{Float64} # Local history for the midpoint height
    h_sq_diff_sum::Vector{Float64}
    h_0_sq_sum::Vector{Float64}
    h_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_A::Vector{Float64}

    function HeightDiffusion(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, buffer_size, zeros(Float64, buffer_size),
            zeros(Float64, num_lags), zeros(Float64, num_lags), 
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end
# Design Note: Both HeightTimeCorr and HeightDiffusion track the history of H_mid.
# For maximum modularity, they each maintain their own history buffer.
# A future optimization could involve a shared height-history workspace if performance
# becomes a critical concern.

get_filename(obs::HeightDiffusion) = "h_diffusion.dat"
get_header(obs::HeightDiffusion) = "t\tHD_Avg\tHD_Err"
get_data(obs::HeightDiffusion) = obs.final_A
get_xaxis(obs::HeightDiffusion, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::HeightDiffusion, sys, workspace)
    current_H_mid = workspace.H_vals[sys.L ÷ 2]
    buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
    obs.H_mid_history[buffer_idx] = current_H_mid

    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_idx = (workspace.t - lag - 1) % obs.buffer_size + 1
            past_H_mid = obs.H_mid_history[past_idx]
            
            # BUG FIX: Corrected typo from h__diff_sum to h_sq_diff_sum
            obs.h_sq_diff_sum[idx] += (current_H_mid - past_H_mid)^2
            obs.h_0_sq_sum[idx] += past_H_mid^2
            obs.h_0_sum[idx] += past_H_mid
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::HeightDiffusion, total_measurements::Int)
    for idx in 1:length(obs.tmea)
        num_meas = obs.num_lag_measurements[idx]
        if num_meas > 0
            avg_h_sq_diff = obs.h_sq_diff_sum[idx] / num_meas
            avg_h_0_sq = obs.h_0_sq_sum[idx] / num_meas
            avg_h_0 = obs.h_0_sum[idx] / num_meas
            denom = avg_h_0_sq - avg_h_0^2
            obs.final_A[idx] = (denom > 1e-9) ? 2.0 - avg_h_sq_diff / denom : 0.0
        end
    end
end

# === Self-registration for ALL observables in this file ===
register_observable!("h_expect", (L, tmea) -> HeightExpect(L))
register_observable!("h_tcorr", (L, tmea) -> HeightTimeCorr(tmea))
register_observable!("h_diffusion", (L, tmea) -> HeightDiffusion(tmea))