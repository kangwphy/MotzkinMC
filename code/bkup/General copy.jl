# File: Observables/Generic.jl
# UPGRADED FOR HIGH PERFORMANCE using Parametric Types

using ..Observables

#==============================================================================
# 1. General Expectation Value
==============================================================================#
mutable struct GeneralExpect <: AbstractObservable
    # Configuration
    filename::String
    header::String
    xaxis::Vector
    quantity_key::Symbol # NEW: The key to request data from the workspace, e.g., :C_vals
    # Data
    sum::Float64
    final_value::Float64

    function GeneralExpect(filename, header, xaxis, quantity_key)
        new(filename, header, xaxis, quantity_key, 0.0, 0.0)
    end
end

get_filename(obs::GeneralExpect) = obs.filename
get_header(obs::GeneralExpect) = obs.header
get_data(obs::GeneralExpect) = [obs.final_value]
get_xaxis(obs::GeneralExpect, L::Int, tmea::Vector{Int}) = obs.xaxis

function measure!(obs::GeneralExpect, sys, workspace)
    # Request the calculated values from the workspace using the key.
    # The workspace handles the lazy calculation.
    op_vector = getproperty(workspace, obs.quantity_key)
    obs.sum += sum(op_vector) / sys.L
end

function finalize!(obs::GeneralExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end



#==============================================================================
# 2. General Time Correlation
==============================================================================#
mutable struct GeneralTCorr{T_OUT <: Number} <: AbstractObservable
    # Configuration
    filename::String
    header::String
    quantity_key::Symbol # NEW: Key for the current-time operator
    # Data
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    # --- Caches for Performance ---
    _history_buffer::Matrix{T_OUT}
    _buffer_size::Int

    function GeneralTCorr(filename, header, tmea, quantity_key::Symbol, L::Int, T_OUT::Type)
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        
        new{T_OUT}(filename, header, quantity_key, tmea,
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags),
            zeros(T_OUT, L, buffer_size), # _history_buffer
            buffer_size)
    end
end

# ... (get_filename, get_header, get_data, get_xaxis are unchanged) ...

get_filename(obs::GeneralTCorr) = obs.filename
get_header(obs::GeneralTCorr) = obs.header
get_data(obs::GeneralTCorr) = obs.final_corr
get_xaxis(obs::GeneralTCorr, L::Int, tmea::Vector{Int}) = obs.tmea

# This function is also now type-stable and fast!
# MODIFIED: measure! now uses in-place updates.
function measure!(obs::GeneralTCorr, sys, workspace)
    # 1. Request operator values for the CURRENT time step from the workspace.
    # This calculation is now shared and cached by the workspace.
    current_op_values = getproperty(workspace, obs.quantity_key)

    # 2. Store this result in our private history buffer.
    buffer_idx = (workspace.t - 1) % obs._buffer_size + 1
    obs._history_buffer[:, buffer_idx] .= current_op_values

    # 3. Loop and correlate using the private history.
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % obs._buffer_size + 1
            past_op_values = view(obs._history_buffer, :, past_buffer_idx)
            
            # Note: We don't need _op_t_cache and _op_0_cache anymore because
            # current_op_values is already a vector we can use.
            obs.prod_sum[idx] += dot(current_op_values, past_op_values) / sys.L
            obs.mean_t_sum[idx] += sum(current_op_values) / sys.L
            obs.mean_0_sum[idx] += sum(past_op_values) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::GeneralTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end