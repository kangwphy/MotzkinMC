# File: periods_config.jl
#
# This file defines the measurement periods for different system sizes (L).
# The key is the system size `L`, and the value is a vector of (threshold, interval) tuples.

const PERIOD_CONFIG = Dict(
    # Parameters for L=100
    100 => [
           (Inf, 1)
    ],

    # Parameters for L=200
    200 => [
        (100000, 1),
        # (10000000, 20),
        (Inf, 20),
    ],

    # Parameters for a very large system, L=1000
    400 => [
        (100000, 1),
        (Inf, 100)
    ],

    800 => [
        (100000, 1),
        (10000000, 100),
        (Inf, 1000),
    ],

    10000 => [
        (10000, 1),
        (1000000, 100),
        (Inf, 1000),
    ]
)

"""
    get_periods_for_L(L::Int, config::Dict)

Finds the appropriate periods parameters for a given system size `L`.

Fallback Logic:
1. If an exact match for `L` is found in the config, it's used.
2. If no exact match, it uses the parameters for the largest `L_config`
   that is smaller than the current `L`.
3. If the current `L` is smaller than all defined `L_config` values,
   it falls back to the parameters of the smallest available `L_config`.
"""
function get_periods(L::Int, config::Dict)
    # Case 0: Config dictionary is empty
    if isempty(config)
        @warn "PERIOD_CONFIG is empty. Defaulting to a single period with interval 1."
        return [(Inf, 1)]
    end
    
    # Case 1: Exact match found
    if haskey(config, L)
        println("INFO: Found exact periods configuration for L = $L.")
        return config[L]
    end

    # Case 2 & 3: No exact match, implement fallback logic
    sorted_Ls = sort(collect(keys(config)))
    
    # Find the largest L in the config that is smaller than our current L
    fallback_L_index = findfirst(k -> k > L, sorted_Ls)
    
    local fallback_L
    if fallback_L_index !== nothing && fallback_L_index > 1
        # Case 2: Found a smaller L to fall back on
        fallback_L = sorted_Ls[fallback_L_index-1]
        @warn "No exact periods config for L = $L. Falling back to settings for nearest smaller size L = $fallback_L."
    elseif fallback_L_index == 1
        fallback_L = sorted_Ls[1]
    else
        # Case 3: Current L is smaller than all keys, use the smallest available config
        fallback_L = sorted_Ls[end]
        @warn "No exact periods config for L = $L. Falling back to settings for smallest available size L = $fallback_L."
    end
    
    return config[fallback_L]
end