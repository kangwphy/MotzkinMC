# File: Observables/Generic.jl
# Contains general-purpose observable structs that are configured by functions.

using ..Observables

#==============================================================================
# 1. General Expectation Value
#    Configured by a predicate function.
==============================================================================#
mutable struct GeneralExpect <: AbstractObservable
    # Configuration
    filename::String
    header::String
    xaxis::Vector
    predicate::Function # The condition to check, e.g., s -> s == m
    # Data
    sum::Float64
    final_value::Float64

    function GeneralExpect(filename, header, xaxis, predicate)
        new(filename, header, xaxis, predicate, 0.0, 0.0)
    end
end

get_filename(obs::GeneralExpect) = obs.filename
get_header(obs::GeneralExpect) = obs.header
get_data(obs::GeneralExpect) = [obs.final_value]
get_xaxis(obs::GeneralExpect, L::Int, tmea::Vector{Int}) = obs.xaxis

function measure!(obs::GeneralExpect, sys, workspace)
    # Apply the predicate function element-wise to the configuration
    p_vector = obs.predicate.(sys.BracketConfig)
    obs.sum += sum(p_vector) / sys.L
end

function finalize!(obs::GeneralExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#==============================================================================
# 2. General Time Correlation
#    Configured by two predicate functions.
==============================================================================#
mutable struct GeneralTCorr <: AbstractObservable
    # Configuration
    filename::String
    header::String
    predicate_t::Function # Predicate for current time
    predicate_0::Function # Predicate for past time
    # Data
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}

    function GeneralTCorr(filename, header, tmea, predicate_t, predicate_0)
        num_lags = length(tmea)
        new(filename, header, predicate_t, predicate_0, tmea,
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::GeneralTCorr) = obs.filename
get_header(obs::GeneralTCorr) = obs.header
get_data(obs::GeneralTCorr) = obs.final_corr
get_xaxis(obs::GeneralTCorr, L::Int, tmea::Vector{Int}) = obs.tmea

function measure!(obs::GeneralTCorr, sys, workspace)
    if !sys.use_history; error("GeneralTCorr requires history."); end
    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)

    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)

            p_t = obs.predicate_t.(current_Z)
            p_0 = obs.predicate_0.(past_Z)
            
            obs.prod_sum[idx] += dot(p_t, p_0) / sys.L
            obs.mean_t_sum[idx] += sum(p_t) / sys.L
            obs.mean_0_sum[idx] += sum(p_0) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::GeneralTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

# === Self-registration ===
# These generic structs are not registered directly.
# Instead, other files will register functions that CREATE instances of these structs.