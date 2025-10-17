# File: Observables/O.jl
# Contains all observables related to the O operator.

using LinearAlgebra
using ..Observables

#--------------------------------------------------------------------
# 1. O Expectation Value
#--------------------------------------------------------------------
mutable struct OExpect <: AbstractObservable
    sum::Float64
    final_value::Float64
    OExpect() = new(0.0, 0.0)
end

get_filename(obs::OExpect) = "o_expect.dat"
get_header(obs::OExpect) = "name\tO_Avg\tO_Err"
get_data(obs::OExpect) = [obs.final_value]
get_xaxis(obs::OExpect, L::Int, tmea::Vector{Int}) = ["O"]

function measure!(obs::OExpect, sys, workspace)
    obs.sum += sum(workspace.O_vals) / sys.L
end

function finalize!(obs::OExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# 2. O-O Spatial Correlation (RCorr)
#--------------------------------------------------------------------
mutable struct OORCorr <: AbstractObservable
    corr_sum::Vector{Float64}
    mean_sum::Vector{Float64}
    final_corr::Vector{Float64}
    function OORCorr(L::Int)
        max_dist = L ÷ 2
        new(zeros(Float64, max_dist), zeros(Float64, L), zeros(Float64, max_dist))
    end
end

get_filename(obs::OORCorr) = "oo_rcorr.dat"
get_header(obs::OORCorr) = "r\tOO_r_Avg\tOO_r_Err"
get_data(obs::OORCorr) = obs.final_corr
get_xaxis(obs::OORCorr, L::Int, tmea::Vector{Int}) = 1:(L÷2)

function measure!(obs::OORCorr, sys, workspace)
    accumulate_spatial_correlation!(obs, workspace.O_vals, sys.L)
end

function finalize!(obs::OORCorr, total_measurements::Int)
    finalize_spatial_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# 3. O-O Time Correlation (TCorr)
#--------------------------------------------------------------------
mutable struct OOTCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    O_history::Matrix{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function OOTCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, buffer_size, Matrix{Int}(undef, 0, 0),
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::OOTCorr) = "oo_tcorr.dat"
get_header(obs::OOTCorr) = "t\tOO_t_Avg\tOO_t_Err"
get_data(obs::OOTCorr) = obs.final_corr
get_xaxis(obs::OOTCorr, L::Int, tmea::Vector{Int}) = tmea

function initialize!(obs::OOTCorr, sys, tmea::Vector{Int})
    obs.O_history = zeros(Int, sys.L, obs.buffer_size)
end

function measure!(obs::OOTCorr, sys, workspace)
    current_O = workspace.O_vals
    current_buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
    @show current_O
    obs.O_history[:, current_buffer_idx] .= current_O
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % obs.buffer_size + 1
            past_O = view(obs.O_history, :, past_buffer_idx)
            obs.prod_sum[idx] += dot(current_O, past_O) / sys.L
            obs.mean_t_sum[idx] += sum(current_O) / sys.L
            obs.mean_0_sum[idx] += sum(past_O) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::OOTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

# === Self-registration ===
register_observable!("o_expect", (L, tmea) -> OExpect())
register_observable!("oo_rcorr", (L, tmea) -> OORCorr(L))
register_observable!("oo_tcorr", (L, tmea) -> OOTCorr(tmea))