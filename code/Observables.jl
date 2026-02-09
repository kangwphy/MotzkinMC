# File: Observables.jl

module Observables

using LinearAlgebra
using Statistics

# export AbstractObservable, register_observable!, load_observables_from_directory!, create_observable
export AbstractObservable, MeasurementWorkspace,
       register_observable!, register_derived_quantity!, load_observables_from_directory!, create_observable,
       register_derived_quantity!, get_values,
       initialize!, measure!, finalize!, get_data, get_xaxis, get_filename, get_header

#==============================================================================
# SECTION 1: Abstract Interface and Registry
==============================================================================#

abstract type AbstractObservable end

# The registry: A dictionary mapping a mode name (String) to a constructor function.
const OBSERVABLE_REGISTRY = Dict{String, Function}()

mutable struct MeasurementWorkspace
    # --- Inputs ---
    const t::Int
    const config::Vector{Int}
    const S::Int

    # --- A single, dynamic cache for all derived quantities ---
    const _cache::Dict{Any, AbstractVector}

    function MeasurementWorkspace(t::Int, config::Vector{Int}, S::Int)
        # Initialize with an empty dictionary.
        new(t, config, S, Dict{Any, AbstractVector}())
    end
end

function _call_calculator(calculator::F, config, S; kwargs...) where {F<:Function}
    return calculator(config, S; kwargs...)
end

function get_values(ws::MeasurementWorkspace, key::Symbol; kwargs...)
    cache_key = isempty(kwargs) ? key : (key, values(kwargs)...)
    if haskey(ws._cache, cache_key); return ws._cache[cache_key]; end
    if haskey(DERIVED_QUANTITY_REGISTRY, key)
        calculator = DERIVED_QUANTITY_REGISTRY[key]
        result = _call_calculator(calculator, ws.config, ws.S; kwargs...)
        ws._cache[cache_key] = result
        return result
    else
        error("No calculator registered for derived quantity ':$key'")
    end
end

function Base.getproperty(ws::MeasurementWorkspace, sym::Symbol)
    if sym in (:t, :config, :S, :_cache)
        return getfield(ws, sym)
    else
        error("Access derived quantity ':$sym' via get_values(workspace, :$sym), not workspace.$sym.")
    end
end

"""
    register_observable!(name::String, constructor)

Registers a new observable by mapping its mode name to its constructor function.
The constructor should be a function that takes (L, tmea) and returns an instance.
"""
function register_observable!(name::String, constructor)
    if haskey(OBSERVABLE_REGISTRY, name)
        @warn "Observable '$name' is already registered. Overwriting."
    end
    OBSERVABLE_REGISTRY[name] = constructor
    # println("Registered observable: '$name'")
end

# NEW: A registry for functions that compute derived quantities like O_vals, H_vals, etc.
const DERIVED_QUANTITY_REGISTRY = Dict{Symbol, Function}()

"""
    register_derived_quantity!(name::Symbol, func::Function)

Registers a function that calculates a derived quantity from the system configuration.
"""
function register_derived_quantity!(name::Symbol, func::Function)
    DERIVED_QUANTITY_REGISTRY[name] = func
    # println("Registered derived quantity calculator: ':$name'")
end

"""
    create_observable(name::String, L::Int, tmea::Vector{Int})

Creates an instance of an observable using its registered mode name.
Returns `nothing` if the name is not found.
"""
function create_observable(name::String, L::Int, tmea::Vector{Int}; kwargs...)
    if !haskey(OBSERVABLE_REGISTRY, name)
        return nothing
    end
    constructor = OBSERVABLE_REGISTRY[name]
    return constructor(L, tmea; kwargs...)
end

"""
    load_observables_from_directory!(dir::String)

Scans a directory for .jl files, includes them, allowing them to self-register.
"""
function load_observables_from_directory!(dir::String)
    # println("\n--- Loading observables from '$(abspath(dir))' ---")
    for filename in readdir(dir)
        if endswith(filename, ".jl")
            include(joinpath(dir, filename))
        end
    end
    # println("--- Observable loading complete ---\n")
end

#==============================================================================
# SECTION 2: Generic Helper Functions
==============================================================================#

function accumulate_spatial_correlation!(obs, source_vector::AbstractVector, L::Int)
    max_dist = L ÷ 2
    obs.mean_sum .+= source_vector
    for r in 1:max_dist
        prod_sum = 0.0
        for i in 1:L
            j = (i + r - 1) % L + 1
            prod_sum += source_vector[i] * source_vector[j]
        end
        obs.corr_sum[r] += prod_sum / L
    end
end

function finalize_spatial_correlation!(obs, total_measurements::Int)
    if total_measurements == 0 return end
    L = length(obs.mean_sum)
    max_dist = length(obs.corr_sum)
    avg_corr_r = obs.corr_sum ./ total_measurements
    mean_val_i = obs.mean_sum ./ total_measurements
    mean_prod = zeros(Float64, max_dist)
    for r in 1:max_dist
        ps = 0.0
        for i in 1:L
            j = (i + r - 1) % L + 1
            ps += mean_val_i[i] * mean_val_i[j]
        end
        mean_prod[r] = ps / L
    end
    obs.final_corr .= avg_corr_r .- mean_prod
end

function finalize_time_correlation!(obs, total_measurements::Int)
    for idx in 1:length(obs.tmea)
        num_meas = obs.num_lag_measurements[idx]
        # @show "final",obs.tmea[idx],num_meas
        if num_meas > 0
            avg_prod = obs.prod_sum[idx] / num_meas
            avg_mean_t = obs.mean_t_sum[idx] / num_meas
            avg_mean_0 = obs.mean_0_sum[idx] / num_meas
            obs.final_corr[idx] = avg_prod - (avg_mean_t * avg_mean_0)
        else
            obs.final_corr[idx] = 0.0
        end
    end
end

#==============================================================================
# SECTION 3: Method Stubs (required for the interface)
==============================================================================#

function initialize! end
function measure! end
function finalize! end
function get_data end
function get_xaxis end
function get_filename end
function get_header end
function initialize!(obs::AbstractObservable, sys, tmea::Vector{Int}) end

end # end of module Observables