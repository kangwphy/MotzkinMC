# File: Observables/Generic.jl
# UPGRADED FOR HIGH PERFORMANCE using Parametric Types

using ..Observables

#==============================================================================
# 1. General Expectation Value
==============================================================================#
# GeneralExpect now stores the key and its parameters (kwargs)
mutable struct GeneralExpect <: AbstractObservable
    filename::String
    header::String
    xaxis::Vector
    quantity_key::Symbol
    quantity_kwargs::NamedTuple
    sum::Float64
    final_value::Float64
    function GeneralExpect(filename, header, xaxis, key, kwargs=NamedTuple())
        new(filename, header, xaxis, key, kwargs, 0.0, 0.0)
    end
end

# Request the calculated values from the workspace using the key.
    # The workspace handles the lazy calculation.
function measure!(obs::GeneralExpect, sys, workspace)
    op_vector = get_values(workspace, obs.quantity_key; obs.quantity_kwargs...)
    obs.sum += sum(op_vector) / sys.L
end



get_filename(obs::GeneralExpect) = obs.filename
get_header(obs::GeneralExpect) = obs.header
get_data(obs::GeneralExpect) = [obs.final_value]
get_xaxis(obs::GeneralExpect, L::Int, tmea::Vector{Int}) = obs.xaxis


function finalize!(obs::GeneralExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end


#==============================================================================
# 2. General Time Correlation (FINAL, CORRECTED CONSTRUCTOR VERSION)
==============================================================================#

mutable struct GeneralTCorr{T_OUT0 <: Number} <: AbstractObservable
    # Configuration
    filename::String
    header::String
    quantity_key_t::Symbol
    quantity_key_0::Symbol
    # Data
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    # Caches
    _history_buffer_0::Matrix{T_OUT0}
    _buffer_size::Int
end

# --- OUTER CONSTRUCTORS (Defined outside the struct block) ---

# Constructor 1: The main constructor for CROSS-correlations <A(t)B(0)>
function GeneralTCorr(filename::String, header::String, tmea::Vector{Int}, key_t::Symbol, key_0::Symbol, L::Int, T_OUT0::Type)
    num_lags = length(tmea)
    buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
    
    # Initialize all accumulator fields
    prod_sum = zeros(Float64, num_lags)
    mean_t_sum = zeros(Float64, num_lags)
    mean_0_sum = zeros(Float64, num_lags)
    num_lag_measurements = zeros(Int, num_lags)
    final_corr = zeros(Float64, num_lags)
    
    # Initialize the history buffer with the correct type and size
    history_buffer = zeros(T_OUT0, L, buffer_size)

    # Call the default inner constructor with all fields specified.
    # We use GeneralTCorr{T_OUT0} to specify the concrete type.
    return GeneralTCorr{T_OUT0}(
        filename, header, key_t, key_0, tmea,
        prod_sum, mean_t_sum, mean_0_sum, num_lag_measurements, final_corr,
        history_buffer, buffer_size
    )
end


#==============================================================================
# 2. General Spatial Correlation (RCorr) - NEW!
==============================================================================#
mutable struct GeneralRCorr <: AbstractObservable
    filename::String
    header::String
    quantity_key::Symbol
    corr_sum::Vector{Float64}
    mean_sum::Vector{Float64}
    final_corr::Vector{Float64}
    function GeneralRCorr(filename, header, key, L::Int)
        max_dist = L ÷ 2
        new(filename, header, key, 
            zeros(Float64, max_dist), zeros(Float64, L), zeros(Float64, max_dist))
    end
end

get_filename(obs::GeneralRCorr) = obs.filename
get_header(obs::GeneralRCorr) = obs.header
get_data(obs::GeneralRCorr) = obs.final_corr
get_xaxis(obs::GeneralRCorr, L::Int, tmea::Vector{Int}) = 1:(L÷2)

function measure!(obs::GeneralRCorr, sys, workspace)
    op_vector = getproperty(workspace, obs.quantity_key)
    accumulate_spatial_correlation!(obs, op_vector, sys.L)
end

function finalize!(obs::GeneralRCorr, total_measurements::Int)
    finalize_spatial_correlation!(obs, total_measurements)
end

# Constructor 3: Convenience constructor for AUTO-correlations <A(t)A(0)>
function GeneralTCorr(filename::String, header::String, tmea::Vector{Int}, key::Symbol, L::Int, T_OUT::Type)
    # This simply calls the main cross-correlation constructor with the same key twice.
    return GeneralTCorr(filename, header, tmea, key, key, L, T_OUT)
end


# --- Methods for GeneralTCorr ---
get_filename(obs::GeneralTCorr) = obs.filename
get_header(obs::GeneralTCorr) = obs.header
get_data(obs::GeneralTCorr) = obs.final_corr
get_xaxis(obs::GeneralTCorr, L::Int, tmea::Vector{Int}) = obs.tmea

function finalize!(obs::GeneralTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

# This is the type-UNSTABLE entry point.
function measure!(obs::GeneralTCorr, sys, workspace)
    # Fetch both operator values. Their types are not known at compile time.
    current_op_values_t = getproperty(workspace, obs.quantity_key_t)
    current_op_values_0 = getproperty(workspace, obs.quantity_key_0)
    
    # Immediately pass them to the type-stable worker function.
    _measure_tcorr_work!(obs, workspace.t, sys.L, current_op_values_t, current_op_values_0)
end

# This is the type-STABLE worker function.
function _measure_tcorr_work!(obs::GeneralTCorr, t::Int, L::Int, current_op_values_t::V, current_op_values_0::V) where {V <: AbstractVector}
    # Store B's result in the history buffer.
    buffer_idx = (t - 1) % obs._buffer_size + 1
    obs._history_buffer_0[:, buffer_idx] .= current_op_values_0

    # Loop and correlate. All operations inside this function are fast.
    for (idx, lag) in enumerate(obs.tmea)
        if t - lag > 0 # Simplified thermalization check, assuming t starts after Tthermal
            past_buffer_idx = (t - lag - 1) % obs._buffer_size + 1
            past_op_values_0 = view(obs._history_buffer_0, :, past_buffer_idx)
            
            obs.prod_sum[idx] += dot(current_op_values_t, past_op_values_0) / L
            obs.mean_t_sum[idx] += sum(current_op_values_t) / L
            obs.mean_0_sum[idx] += sum(past_op_values_0) / L
            obs.num_lag_measurements[idx] += 1
        end
    end
end