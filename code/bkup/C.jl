# File: Observables/C.jl
# Contains all observables related to the C operator.

using LinearAlgebra
using ..Observables

#--------------------------------------------------------------------
# 1. C Expectation Value
#--------------------------------------------------------------------
mutable struct CExpect <: AbstractObservable
    sum::Float64
    final_value::Float64
    CExpect() = new(0.0, 0.0)
end

get_filename(obs::CExpect) = "c_expect.dat"
get_header(obs::CExpect) = "Observable\tAvg\tErr"
get_data(obs::CExpect) = [obs.final_value]
get_xaxis(obs::CExpect, L::Int, tmea::Vector{Int}) = ["<C>"]

function measure!(obs::CExpect, sys, workspace)
    obs.sum += sum(workspace.C_vals) / sys.L
end

function finalize!(obs::CExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# 2. C-C Time Correlation (TCorr)
#--------------------------------------------------------------------
mutable struct CCTCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    C_history::Matrix{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function CCTCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, buffer_size, Matrix{Int}(undef, 0, 0),
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::CCTCorr) = "cc_tcorr.dat"
get_header(obs::CCTCorr) = "t\tCC_t_Avg\tCC_t_Err"
get_data(obs::CCTCorr) = obs.final_corr
get_xaxis(obs::CCTCorr, L::Int, tmea::Vector{Int}) = tmea

function initialize!(obs::CCTCorr, sys, tmea::Vector{Int})
    obs.C_history = zeros(Int, sys.L, obs.buffer_size)
end

function measure!(obs::CCTCorr, sys, workspace)
    current_C = workspace.C_vals
    current_buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
    obs.C_history[:, current_buffer_idx] .= current_C
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % obs.buffer_size + 1
            past_C = view(obs.C_history, :, past_buffer_idx)
            obs.prod_sum[idx] += dot(current_C, past_C) / sys.L
            obs.mean_t_sum[idx] += sum(current_C) / sys.L
            obs.mean_0_sum[idx] += sum(past_C) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::CCTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

# === Self-registration ===
register_observable!("c_expect", (L, tmea) -> CExpect())
register_observable!("cc_tcorr", (L, tmea) -> CCTCorr(tmea))