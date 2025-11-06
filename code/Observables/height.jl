# File: Observables/Height.jl
# Contains all observables related to the Height operator (h).
# This file defines its own concrete structs for specialized measurements
# (profiles and single-point correlations) but registers H_vals as a derived quantity.

using ..Observables

#==============================================================================
# 1. Calculation Logic for the H operator
==============================================================================#

const H_KEY = :H_vals

# This function should be MOVED here from sub.jl
function height_profile!(h_vals::AbstractVector, config::Vector{Int})
    current_height = 0
    @inbounds for i in 1:length(config)
        current_height += config[i]
        h_vals[i] = current_height
    end
    return h_vals
end

# Register H_vals as a "derived quantity" that the workspace can compute on demand.
register_derived_quantity!(H_KEY, (config, S) -> begin
    Hvals = zeros(Int, length(config))
    height_profile!(Hvals, config)
    return Hvals
end)


#==============================================================================
# 2. Concrete Observable Implementations for Height
==============================================================================#

#--------------------------------------------------------------------
# 2.1 Height Expectation Profile <h(i)>
# (This is a profile measurement, so it uses a concrete struct, not GeneralExpect)
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
    # This now lazily computes H_vals on the first request in a time step
    h_vals = get_values(workspace, H_KEY)
    _measure_height_expect_work!(obs, h_vals)
end

# Type-stable worker function
function _measure_height_expect_work!(obs::HeightExpect, h_vals::V) where {V <: AbstractVector}
    obs.mean_sum .+= h_vals
end

function finalize!(obs::HeightExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value .= obs.mean_sum ./ total_measurements
    end
end

#--------------------------------------------------------------------
# 2.2 Height-Height Time Correlation at the center point L/2
# (This is a single-point correlation, not a vector one, so it needs a concrete struct)
#--------------------------------------------------------------------
mutable struct HeightTimeCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    H_mid_history::Vector{Float64}
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

get_filename(obs::HeightTimeCorr) = "hh_tcorr.dat"
get_header(obs::HeightTimeCorr) = "t\thh_t_Avg\thh_t_Err"
get_data(obs::HeightTimeCorr) = obs.final_corr
get_xaxis(obs::HeightTimeCorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::HeightTimeCorr, sys, workspace)
    # Get the full height profile from the workspace (computed on demand)
    h_vals = get_values(workspace, H_KEY)
    _measure_height_tcorr_work!(obs, h_vals, workspace.t, sys.L, sys.Tthermal)
end

# Type-stable worker function
function _measure_height_tcorr_work!(obs::HeightTimeCorr, h_vals::V, t::Int, L::Int, Tthermal::Int) where {V <: AbstractVector}
    current_H_mid = h_vals[L ÷ 2]
    
    buffer_idx = (t - 1) % obs.buffer_size + 1
    obs.H_mid_history[buffer_idx] = current_H_mid

    for (idx, lag) in enumerate(obs.tmea)
        if t - lag > Tthermal
            past_idx = (t - lag - 1) % obs.buffer_size + 1
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
# 2.3 Height Diffusion A(t)
# (This has a unique finalization logic, so it needs a concrete struct)
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
# ... (constructor, get_filename, etc. are unchanged) ...

get_filename(obs::HeightDiffusion) = "hd_tcorr.dat"
get_header(obs::HeightDiffusion) = "t\thd_t_Avg\thd_t_Err"
get_data(obs::HeightDiffusion) = obs.final_A
get_xaxis(obs::HeightDiffusion, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::HeightDiffusion, sys, workspace)
    h_vals = get_values(workspace, H_KEY)
    _measure_height_diffusion_work!(obs, h_vals, workspace.t, sys.L, sys.Tthermal)
end

# Type-stable worker function
function _measure_height_diffusion_work!(obs::HeightDiffusion, h_vals::V, t::Int, L::Int, Tthermal::Int) where {V <: AbstractVector}
    current_H_mid = h_vals[L ÷ 2]
    buffer_idx = (t - 1) % obs.buffer_size + 1
    obs.H_mid_history[buffer_idx] = current_H_mid

    for (idx, lag) in enumerate(obs.tmea)
        if t - lag > Tthermal
            past_idx = (t - lag - 1) % obs.buffer_size + 1
            past_H_mid = obs.H_mid_history[past_idx]
            
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


#==============================================================================
# 3. Self-registration for ALL observables in this file
==============================================================================#

register_observable!("h_expect", (L, tmea; kwargs...) -> HeightExpect(L))
register_observable!("hh_tcorr", (L, tmea; kwargs...) -> HeightTimeCorr(tmea))
register_observable!("hd_tcorr", (L, tmea; kwargs...) -> HeightDiffusion(tmea))