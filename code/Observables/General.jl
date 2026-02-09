# File: Observables/Generic.jl
# UPGRADED FOR HIGH PERFORMANCE using Parametric Types

using ..Observables

#==============================================================================
# 1. General Expectation Value
==============================================================================#
# GeneralExpect now stores the key and its parameters (kwargs)
mutable struct GeneralExpect{KW <: NamedTuple} <: AbstractObservable
    filename::String
    header::String
    xaxis::Vector
    quantity_key::Symbol
    key_kwargs::KW
    sum::Float64
    final_value::Float64
    function GeneralExpect(filename, header, xaxis, key; key_kwargs::KW = NamedTuple()) where {KW <: NamedTuple}
        new{KW}(filename, header, xaxis, key, key_kwargs, 0.0, 0.0)
    end
end

get_filename(obs::GeneralExpect) = obs.filename
get_header(obs::GeneralExpect) = obs.header
get_data(obs::GeneralExpect) = [obs.final_value]
get_xaxis(obs::GeneralExpect, L::Int, tmea::Vector{Int}) = obs.xaxis

function measure!(obs::GeneralExpect, sys, workspace)
    # Request the calculated values from the workspace using the key.
    # The workspace handles the lazy calculation.
    op_vector = get_values(workspace, obs.quantity_key; obs.key_kwargs...)
    obs.sum += sum(op_vector) / sys.L
end

function finalize!(obs::GeneralExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
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
    op_vector = get_values(workspace, obs.quantity_key)
    accumulate_spatial_correlation!(obs, op_vector, sys.L)
end

function finalize!(obs::GeneralRCorr, total_measurements::Int)
    finalize_spatial_correlation!(obs, total_measurements)
end



#==============================================================================
# 2. General Time Correlation (FINAL, CORRECTED CONSTRUCTOR VERSION)
==============================================================================#

mutable struct GeneralTCorr{T_OUT0 <: Number, KW_T <: NamedTuple, KW_0 <: NamedTuple} <: AbstractObservable
    filename::String
    header::String
    quantity_key_t::Symbol
    quantity_kwargs_t::KW_T
    quantity_key_0::Symbol
    quantity_kwargs_0::KW_0
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
function GeneralTCorr(filename::String, header::String, tmea::Vector{Int}, key_t::Symbol, key_0::Symbol, L::Int, T_OUT0::Type; kwargs_t::NamedTuple = NamedTuple(), kwargs_0::NamedTuple = NamedTuple())
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
    return GeneralTCorr{T_OUT0,typeof(kwargs_t), typeof(kwargs_0)}(
        filename, header, key_t,  kwargs_t, key_0, kwargs_0, tmea,
        prod_sum, mean_t_sum, mean_0_sum, num_lag_measurements, final_corr,
        history_buffer, buffer_size
    )
end
# Constructor 3: Convenience constructor for AUTO-correlations <A(t)A(0)>
function GeneralTCorr(filename::String, header::String, tmea::Vector{Int}, key::Symbol, L::Int, T_OUT::Type; kwargs_t::NamedTuple = NamedTuple(), kwargs_0::NamedTuple = NamedTuple())
    # This simply calls the main cross-correlation constructor with the same key twice.
    return GeneralTCorr(filename, header, tmea, key, key, L, T_OUT; kwargs_t=kwargs_t, kwargs_0=kwargs_0)
end


# --- Methods for GeneralTCorr ---
get_filename(obs::GeneralTCorr) = obs.filename
get_header(obs::GeneralTCorr) = obs.header
get_data(obs::GeneralTCorr) = obs.final_corr
get_xaxis(obs::GeneralTCorr, L::Int, tmea::Vector{Int}) = tmea

function finalize!(obs::GeneralTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

# This is the type-UNSTABLE entry point.
function measure!(obs::GeneralTCorr, sys, workspace)
    # Fetch both operator values. Their types are not known at compile time.
    current_op_values_t = get_values(workspace, obs.quantity_key_t; obs.quantity_kwargs_t...)
    current_op_values_0 = get_values(workspace, obs.quantity_key_0; obs.quantity_kwargs_0...)
    
    # Immediately pass them to the type-stable worker function.
    _measure_tcorr_work!(obs, workspace.t, sys.L, sys.Tthermal, current_op_values_t, current_op_values_0)
end

# This is the type-STABLE worker function.
function _measure_tcorr_work!(obs::GeneralTCorr, t::Int, L::Int, Tthermal::Int, current_op_values_t::V, current_op_values_0::V) where {V <: AbstractVector}
    # Store B's result in the history buffer.
    buffer_idx = (t - 1) % obs._buffer_size + 1
    obs._history_buffer_0[:, buffer_idx] .= current_op_values_0

    # Loop and correlate. All operations inside this function are fast.
    for (idx, lag) in enumerate(obs.tmea)
        if t - lag > Tthermal # Simplified thermalization check, assuming t starts after Tthermal
            past_buffer_idx = (t - lag - 1) % obs._buffer_size + 1
            past_op_values_0 = view(obs._history_buffer_0, :, past_buffer_idx)
            
            # --- DEBUG: Check if past buffer slot was actually written ---
            past_sum = sum(past_op_values_0)
            curr_sum = sum(current_op_values_0)
            if obs.num_lag_measurements[idx] < 5  # Only print first few
                println("[DEBUG TCorr] t=$t, lag=$lag, past_t=$(t-lag), " *
                        "buffer_idx=$buffer_idx, past_buffer_idx=$past_buffer_idx, " *
                        "sum(past)=$past_sum, sum(current)=$curr_sum, " *
                        "all_zero_past=$(all(x->x==0, past_op_values_0)), " *
                        "num_meas_so_far=$(obs.num_lag_measurements[idx])")
            end
            # --- END DEBUG ---

            obs.prod_sum[idx] += dot(current_op_values_t, past_op_values_0) / L
            obs.mean_t_sum[idx] += sum(current_op_values_t) / L
            obs.mean_0_sum[idx] += sum(past_op_values_0) / L
            obs.num_lag_measurements[idx] += 1
        end
    end
end