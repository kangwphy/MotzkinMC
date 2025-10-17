# File: Observables/Generic.jl
# UPGRADED FOR HIGH PERFORMANCE using Parametric Types

using ..Observables

#==============================================================================
# 1. General Expectation Value
#    Now parameterized by the function type F for performance.
==============================================================================#
mutable struct GeneralExpect{F <: Function} <: AbstractObservable
    # Configuration
    filename::String
    header::String
    xaxis::Vector
    op_func::F # The field now has a CONCRETE type F
    # Data
    sum::Float64
    final_value::Float64

    function GeneralExpect(filename, header, xaxis, op_func::F) where {F <: Function}
        new{F}(filename, header, xaxis, op_func, 0.0, 0.0)
    end
end

get_filename(obs::GeneralExpect) = obs.filename
get_header(obs::GeneralExpect) = obs.header
get_data(obs::GeneralExpect) = [obs.final_value]
get_xaxis(obs::GeneralExpect, L::Int, tmea::Vector{Int}) = obs.xaxis

# This function is now type-stable and fast!
function measure!(obs::GeneralExpect, sys, workspace)
    # The compiler knows the exact type of obs.op_func at compile time.
    op_vector = obs.op_func.(sys.BracketConfig)
    obs.sum += sum(op_vector) / sys.L
end

function finalize!(obs::GeneralExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

# ... (GeneralExpect 保持不变) ...
#==============================================================================
# 2. General Time Correlation (FINAL, HIGH-PERFORMANCE VERSION)
==============================================================================#
mutable struct GeneralTCorr{F1 <: Function, F2 <: Function} <: AbstractObservable
    # Configuration
    filename::String
    header::String
    op_func_t::F1
    op_func_0::F2
    # Data
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    # --- Caches for Performance ---
    _op_t_cache::Vector      # Cache for current operator values
    _op_0_cache::Vector      # Cache for past operator values
    _history_buffer::Matrix  # NEW: Cache for operator values over time
    _buffer_size::Int

    function GeneralTCorr(filename, header, tmea, op_func_t::F1, op_func_0::F2, L::Int) where {F1, F2}
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        
        new{F1, F2}(filename, header, op_func_t, op_func_0, tmea,
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags),
            zeros(Int, L), # op_t_cache
            zeros(Int, L), # op_0_cache
            zeros(Int, L, buffer_size), # _history_buffer
            buffer_size)
    end
end

# ... (get_filename, get_header, get_data, get_xaxis are unchanged) ...

get_filename(obs::GeneralTCorr) = obs.filename
get_header(obs::GeneralTCorr) = obs.header
get_data(obs::GeneralTCorr) = obs.final_corr
get_xaxis(obs::GeneralTCorr, L::Int, tmea::Vector{Int}) = obs.tmea

# FINAL, HIGH-PERFORMANCE version of measure!
function measure!(obs::GeneralTCorr, sys, workspace)
    # 1. Calculate operator values for the CURRENT time step ONCE.
    current_op_values = obs._op_t_cache # Use the pre-allocated cache
    current_op_values .= obs.op_func_t.(sys.BracketConfig)

    # 2. Store this result in the history buffer.
    buffer_idx = (workspace.t - 1) % obs._buffer_size + 1
    obs._history_buffer[:, buffer_idx] .= current_op_values

    # 3. Loop through time lags and correlate.
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % obs._buffer_size + 1
            
            # 4. RETRIEVE pre-calculated past values directly from history. NO re-computation!
            past_op_values = view(obs._history_buffer, :, past_buffer_idx)
            
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