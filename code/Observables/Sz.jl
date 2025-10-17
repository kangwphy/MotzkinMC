# File: Observables/Sz.jl
# Contains all observables related to the Sz operator.

using LinearAlgebra
using ..Observables

#--------------------------------------------------------------------
# 1. Sz Expectation Value
#--------------------------------------------------------------------
mutable struct SzExpect <: AbstractObservable
    mean_sum::Vector{Float64}
    final_value::Vector{Float64}
    SzExpect(L::Int) = new(zeros(Float64, L), zeros(Float64, L))
end

get_filename(obs::SzExpect) = "sz_expect.dat"
get_header(obs::SzExpect) = "Site\tSz_Avg\tSz_Err"
get_data(obs::SzExpect) = obs.final_value
get_xaxis(obs::SzExpect, L::Int, tmea::Vector{Int}) = 1:L

function measure!(obs::SzExpect, sys, workspace)
    obs.mean_sum .+= sys.BracketConfig
end

function finalize!(obs::SzExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value .= obs.mean_sum ./ total_measurements
    end
end

#--------------------------------------------------------------------
# 2. Sz-Sz Spatial Correlation (RCorr)
#--------------------------------------------------------------------
mutable struct SzSzRCorr <: AbstractObservable
    corr_sum::Vector{Float64}
    mean_sum::Vector{Float64}
    final_corr::Vector{Float64}
    function SzSzRCorr(L::Int)
        max_dist = L ÷ 2
        new(zeros(Float64, max_dist), zeros(Float64, L), zeros(Float64, max_dist))
    end
end

get_filename(obs::SzSzRCorr) = "szsz_rcorr.dat"
get_header(obs::SzSzRCorr) = "r\tSzSz_r_Avg\tSzSz_r_Err"
get_data(obs::SzSzRCorr) = obs.final_corr
get_xaxis(obs::SzSzRCorr, L::Int, tmea::Vector{Int}) = 1:(L÷2)

function measure!(obs::SzSzRCorr, sys, workspace)
    accumulate_spatial_correlation!(obs, sys.BracketConfig, sys.L)
end

function finalize!(obs::SzSzRCorr, total_measurements::Int)
    finalize_spatial_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# 3. Sz-Sz Time Correlation (TCorr)
#--------------------------------------------------------------------
mutable struct SzSzTCorr <: AbstractObservable
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function SzSzTCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        new(tmea, zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::SzSzTCorr) = "szsz_tcorr.dat"
get_header(obs::SzSzTCorr) = "t\tSzSz_t_Avg\tSzSz_t_Err"
get_data(obs::SzSzTCorr) = obs.final_corr
get_xaxis(obs::SzSzTCorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::SzSzTCorr, sys, workspace)
    if !sys.use_history
        error("SzSzTCorr requires the system's history buffer to be enabled.")
    end
    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)
            obs.prod_sum[idx] += dot(current_Z, past_Z) / sys.L
            obs.mean_t_sum[idx] += sum(current_Z) / sys.L
            obs.mean_0_sum[idx] += sum(past_Z) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::SzSzTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

# === Self-registration ===
register_observable!("sz_expect", (L, tmea) -> SzExpect(L))
register_observable!("szsz_rcorr", (L, tmea) -> SzSzRCorr(L))
register_observable!("szsz_tcorr", (L, tmea) -> SzSzTCorr(tmea))