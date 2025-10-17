# File: Observables/O.jl
# Contains all observables related to the O operator.

using LinearAlgebra
using ..Observables

#--------------------------------------------------------------------
# 1. Q Expectation Value
#--------------------------------------------------------------------
mutable struct QExpect <: AbstractObservable
    sum::Float64
    final_value::Float64
    QExpect() = new(0.0, 0.0)
end

get_filename(obs::QExpect) = "q_expect.dat"
get_header(obs::QExpect) = "name\tQ_Avg\tQ_Err"
get_data(obs::QExpect) = [obs.final_value]
get_xaxis(obs::QExpect, L::Int, tmea::Vector{Int}) = ["Q"]

function measure!(obs::QExpect, sys, workspace)
    obs.sum += sum(workspace.Q_vals) / sys.L
end

function finalize!(obs::QExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# 2. O-O Spatial Correlation (RCorr)
#--------------------------------------------------------------------
mutable struct QQRCorr <: AbstractObservable
    corr_sum::Vector{Float64}
    mean_sum::Vector{Float64}
    final_corr::Vector{Float64}
    function QQRCorr(L::Int)
        max_dist = L ÷ 2
        new(zeros(Float64, max_dist), zeros(Float64, L), zeros(Float64, max_dist))
    end
end

get_filename(obs::QQRCorr) = "qq_rcorr.dat"
get_header(obs::QQRCorr) = "r\tQQ_r_Avg\tQQ_r_Err"
get_data(obs::QQRCorr) = obs.final_corr
get_xaxis(obs::QQRCorr, L::Int, tmea::Vector{Int}) = 1:(L÷2)

function measure!(obs::QQRCorr, sys, workspace)
    accumulate_spatial_correlation!(obs, workspace.O_vals, sys.L)
end

function finalize!(obs::QQRCorr, total_measurements::Int)
    finalize_spatial_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# 3. Q-Q Time Correlation (TCorr)
#--------------------------------------------------------------------
mutable struct QQTCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    O_history::Matrix{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function QQTCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, buffer_size, Matrix{Int}(undef, 0, 0),
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::QQTCorr) = "qq_tcorr.dat"
get_header(obs::QQTCorr) = "t\tQQ_t_Avg\tQQ_t_Err"
get_data(obs::QQTCorr) = obs.final_corr
get_xaxis(obs::QQTCorr, L::Int, tmea::Vector{Int}) = tmea

function initialize!(obs::QQTCorr, sys, tmea::Vector{Int})
    obs.O_history = zeros(Int, sys.L, obs.buffer_size)
end

function measure!(obs::QQTCorr, sys, workspace)
    current_O = workspace.O_vals
    current_buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
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

function finalize!(obs::QQTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

# === Self-registration ===
register_observable!("q_expect", (L, tmea) -> QExpect())
register_observable!("qq_rcorr", (L, tmea) -> QQRCorr(L))
register_observable!("qq_tcorr", (L, tmea) -> QQTCorr(tmea))