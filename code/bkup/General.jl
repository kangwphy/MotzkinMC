# File: Observables/Generic.jl
# UPGRADED FOR HIGH PERFORMANCE using Parametric Types
##v3.0
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
# 2. General Time Correlation
#    UPGRADED: Now with pre-allocated caches to avoid allocations in the loop.
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
    # --- NEW: Pre-allocated workspace for measure! ---
    _op_t_cache::Vector{Float64}
    _op_0_cache::Vector{Float64}

    # MODIFIED: Constructor now accepts L to pre-allocate caches.
    function GeneralTCorr(filename, header, tmea, op_func_t::F1, op_func_0::F2, L::Int) where {F1 <: Function, F2 <: Function}
        num_lags = length(tmea)
        # Pre-allocate caches of the correct size
        op_t_cache = zeros(Float64, L)
        op_0_cache = zeros(Float64, L)
        new{F1, F2}(filename, header, op_func_t, op_func_0, tmea,
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags),
            op_t_cache, op_0_cache) # Add new fields
    end
end

get_filename(obs::GeneralTCorr) = obs.filename
get_header(obs::GeneralTCorr) = obs.header
get_data(obs::GeneralTCorr) = obs.final_corr
get_xaxis(obs::GeneralTCorr, L::Int, tmea::Vector{Int}) = obs.tmea

# This function is also now type-stable and fast!
# MODIFIED: measure! now uses in-place updates.
function measure!(obs::GeneralTCorr, sys, workspace)
    if !sys.use_history; error("GeneralTCorr requires history."); end
    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)
    obs._op_t_cache .= obs.op_func_t.(current_Z)
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)

            # Use in-place broadcasting (.=) to update pre-allocated caches.
            # This avoids creating new arrays.
            # obs._op_t_cache = obs.op_func_t.(current_Z)
            # obs._op_0_cache = obs.op_func_0.(past_Z)
            # obs._op_t_cache = abs.(current_Z) .== 1
            # obs._op_0_cache = abs.(past_Z) .== 1

            
            obs._op_0_cache .= obs.op_func_0.(past_Z)
            
            # Perform calculations using the cached vectors
            obs.prod_sum[idx] += dot(current_Z, past_Z) / sys.L
            obs.mean_t_sum[idx] += sum(current_Z) / sys.L
            obs.mean_0_sum[idx] += sum(past_Z) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::GeneralTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end