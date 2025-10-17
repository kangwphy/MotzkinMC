using Random, Dates, Serialization
using LinearAlgebra
include("initial.jl")
# include("Observables.jl") # Include the new observables file

# The BracketSystem is now lean, only holding the system's physical state.
mutable struct BracketSystem
    # --- System Parameters ---
    S::Int
    L::Int
    T::Int
    Tthermal::Int
    seed::Int
    rank::Int
    
    # --- System State ---
    BracketConfig::Vector{Int}

    # --- Shared History Buffer (for time-displaced observables) ---
    # This is a shared resource for any observable that needs it,
    # to avoid each one storing a full copy of the system history.
    use_history::Bool
    buffer_size::Int
    Z_history::Matrix{Int}
end

function BracketSystem(L::Int, T::Int, Tthermal::Int, tmea::Vector{Int}, seed::Int, rank::Int, S::Int, use_history::Bool)
    Random.seed!(seed)
    BracketConfig = zeros(Int, L)
    
    buffer_size = use_history ? (isempty(tmea) ? 1 : maximum(tmea) + 1) : 0
    Z_history = use_history ? zeros(Int, L, buffer_size) : Matrix{Int}(undef, 0, 0)

    return BracketSystem(S, L, T, Tthermal, seed, rank,
                         BracketConfig,
                         use_history, buffer_size, Z_history)
end

"""
Initializes a configuration for a Spin-S Motzkin chain.
"""
function InitialConfigSpinS!(sys::BracketSystem)
    S = sys.S
    if S == 1
        n = (sys.L - 2) ÷ 3
        left, right = n, n

        for i in 1:sys.L
            ticket = rand()
            remaining = sys.L - (i - 1)
            if ticket < left / remaining
                sys.BracketConfig[i] = 1 # Left bracket
                left -= 1
            elseif ticket < (left + right) / remaining
                sys.BracketConfig[i] = -1 # Right bracket
                right -= 1
            else
                sys.BracketConfig[i] = 0
            end
        end
    else
        sys.BracketConfig = generate_initial_state(sys.L, sys.S, method="random")
    end
    if sum(sys.BracketConfig) != 0
        @warn "Initial configuration has non-zero total spin! sum=$(sum(sys.BracketConfig))"
    end
end


# # A temporary container for data calculated once per time-step to be shared among observables.
# struct MeasurementWorkspace
#     t::Int
#     O_vals::Vector{Int}
#     H_vals::Vector{Int}
# end

using Base: @kwdef

# An intelligent, lazy-loading workspace for measurement data.
# mutable struct MeasurementWorkspace
#     # --- Inputs ---
#     const t::Int
#     const config::Vector{Int}
#     const S::Int

#     # --- Caches (internal fields) ---
#     # These store the results once calculated for the current time step.
#     # O_vals::Union{Vector{Int}, Nothing}
#     # _Q_vals::Union{Vector{Int}, Nothing}
#     # _H_vals::Union{Vector{Int}, Nothing}
#     # _C_vals::Union{Vector{Int}, Nothing}

#     # function MeasurementWorkspace(t::Int, config::Vector{Int}, S::Int)
#     #     new(t, config, S, nothing, nothing, nothing)
#     #     # new(t, config, S, nothing, nothing, nothing, nothing)
#     # end
# end

# NEW: Add a "Function Barrier" helper function.
# This function captures the concrete type `F` of the calculator.
mutable struct MeasurementWorkspace
    # --- Inputs ---
    const t::Int
    const config::Vector{Int}
    const S::Int

    # --- A single, dynamic cache for all derived quantities ---
    const _cache::Dict{Symbol, AbstractVector}

    function MeasurementWorkspace(t::Int, config::Vector{Int}, S::Int)
        # Initialize with an empty dictionary.
        new(t, config, S, Dict{Symbol, AbstractVector}())
    end
end


function _call_calculator(calculator::F, config, S; kwargs...) where {F<:Function}
    return calculator(config, S; kwargs...)
end


# NEW: The official, type-stable way to get derived quantities
function get_values(ws::MeasurementWorkspace, key::Symbol; kwargs...)
    # 1. Create a unique cache key from the base key and its parameters
    cache_key = isempty(kwargs) ? key : (key, values(kwargs)...)

    # 2. Check if it's already in the cache
    if haskey(ws._cache, cache_key)
        return ws._cache[cache_key]
    end

    # 3. If not, find the calculator and compute the value
    if haskey(Observables.DERIVED_QUANTITY_REGISTRY, key)
        calculator = Observables.DERIVED_QUANTITY_REGISTRY[key]
        result = _call_calculator(calculator, ws.config, ws.S; kwargs...)
        ws._cache[cache_key] = result
        return result
    else
        error("No calculator registered for derived quantity ':$key'")
    end
end

# Base.getproperty is now simplified for only internal fields.
function Base.getproperty(ws::MeasurementWorkspace, sym::Symbol)
    if sym in (:t, :config, :S, :_cache)
        return getfield(ws, sym)
    else
        # All derived quantities should now be accessed via get_values()
        error("Accessing derived quantity ':$sym' directly is deprecated. Use get_values(workspace, :$sym).")
    end
end


# # This function overloads what happens when you access `workspace.field`.
# # It is now fully generic and works with the dictionary cache.
# function Base.getproperty(ws::MeasurementWorkspace, sym::Symbol)
#     # If the property is a standard field, return it directly.
#     if sym in (:t, :config, :S, :_cache)
#         return getfield(ws, sym)
#     end

#     # Otherwise, `sym` is a derived quantity (e.g., :C_vals) that we need to compute or get from cache.
    
#     # 1. Check if it's already in the cache for this time step.
#     if haskey(ws._cache, sym)
#         return ws._cache[sym]
#     end

#     # 2. If not in cache, check if there is a registered calculator for it.
#     if haskey(Observables.DERIVED_QUANTITY_REGISTRY, sym)
#         # Get the calculator function from the registry
#         calculator = Observables.DERIVED_QUANTITY_REGISTRY[sym]
        
#         # Call the calculator (using the function barrier for performance)
#         result = _call_calculator(calculator, ws.config, ws.S)
        
#         # 3. Store the result in the cache for future use in this time step.
#         ws._cache[sym] = result
        
#         return result
#     else
#         # If no calculator is registered, it's an error.
#         error("No calculator registered for derived quantity ':$sym', and it's not a standard field of MeasurementWorkspace.")
#     end
# end

# # This function overloads what happens when you access `workspace.field`.
# function Base.getproperty(ws::MeasurementWorkspace, sym::Symbol)
#     # Check if the requested symbol is a cacheable derived quantity
#     cache_sym = Symbol("_", sym) # e.g., :C_vals -> :_C_vals
    
#     if hasfield(MeasurementWorkspace, cache_sym)
#         # If the cache is already populated, return it
#         if getfield(ws, cache_sym) !== nothing
#             return getfield(ws, cache_sym)
#         end
        
#         # Cache is empty, so we need to compute it.
#         if haskey(Observables.DERIVED_QUANTITY_REGISTRY, sym)
#             calculator = Observables.DERIVED_QUANTITY_REGISTRY[sym]
#             result = calculator(ws.config, ws.S)
#             setfield!(ws, cache_sym, result) # Store in cache, e.g., ws._C_vals = result
#             return result
#         else
#             error("No calculator registered for derived quantity ':$sym'")
#         end
#     else
#         # For any other field (like :t or :config), return it directly.
#         return getfield(ws, sym)
#     end
# end
# function Base.getproperty(ws::MeasurementWorkspace, sym::Symbol)
#     # For :H_vals, calculate it on demand
#     if sym === :H_vals
#         if ws._H_vals === nothing
#             # If the cache is empty, calculate it now and store it.
#             # println("DEBUG: Calculating H_vals for the first time at t=$(ws.t)") # Optional debug print
#             ws._H_vals = zeros(Int, length(ws.config))
#             height_profile!(ws._H_vals, ws.config)
#         end
#         return ws._H_vals
    
#     # For :O_vals, calculate it on demand
#     elseif sym === :O_vals
#         if ws._O_vals === nothing
#             # If the cache is empty, calculate it now and store it.
#             # println("DEBUG: Calculating O_vals for the first time at t=$(ws.t)") # Optional debug print
#             ws._O_vals = zeros(Int, length(ws.config))
#             MeasureO_SpinS!(ws._O_vals, ws.config, ws.S)
#         end
#         return ws._O_vals
#     elseif sym === :Q_vals
#         if ws._Q_vals === nothing
#             # If the cache is empty, calculate it now and store it.
#             # println("DEBUG: Calculating O_vals for the first time at t=$(ws.t)") # Optional debug print
#             ws._Q_vals = zeros(Int, length(ws.config))
#             MeasureQ_SpinS!(ws._Q_vals, ws.config, ws.S)
#         end
#         return ws._Q_vals

#     # # For :C_vals, calculate it on demand
#     # elseif sym === :C_vals
#     #     if ws._C_vals === nothing
#     #         # println("DEBUG: Calculating C_vals for the first time at t=$(ws.t)") # Optional debug print
#     #         ws._C_vals = zeros(Int, length(ws.config))
#     #         MeasureC!(ws._C_vals, ws.config)
#     #     end
#     #     return ws._C_vals

#     # For any other field (like :t), just return it directly.
#     else
#         return getfield(ws, sym)
#     end
# end

"""
Measures the O operator for the generalized Spin-S model.
O = ∑ₖ|dᵏuᵏ⟩⟨00|, where k runs from 1 to S.
Classical Oc = ∑ₖ(|dᵏuᵏ⟩⟨dᵏuᵏ| + |00⟩⟨00|).
"""
function MeasureO_SpinS!(Ovals::Vector{Int}, config::Vector{Int}, S::Int)
    L = length(config)
    @inbounds for i in 1:L
        j = (i % L) + 1
        l, r = config[i], config[j]
        Ovals[i] = (l == 0 && r == 0) ? S : ((l < 0 && l == -r) ? 1 : 0)  # ∑ₖ|dᵏuᵏ⟩⟨00|
    end
    return Ovals
end
"""
Measures the O operator for the generalized Spin-S model.
O = ∑ₖ|dᵏuᵏ⟩⟨00|, where k runs from 1 to S.
Classical Oc = ∑ₖ(|uᵏdᵏ⟩⟨uᵏdᵏ| + |00⟩⟨00|).
"""
function MeasureQ_SpinS!(Qvals::Vector{Int}, config::Vector{Int}, S::Int)
    L = length(config)
    @inbounds for i in 1:L
        j = (i % L) + 1
        l, r = config[i], config[j]
        Qvals[i] = (l == 0 && r == 0) ? S : ((l > 0 && l == -r) ? 1 : 0)  # ∑ₖ|uᵏdᵏ⟩⟨00|
    end
    return Qvals
end
"""
Calculates the height profile h(x) from a spin configuration Sz(x).
h(x) = sum_{i=1 to x} Sz(i).
"""
function height_profile!(h_vals::AbstractVector{Int}, config::Vector{Int})
    current_height = 0
    @inbounds for i in 1:length(config)
        current_height += config[i]
        h_vals[i] = current_height
    end
    return h_vals
end

function MeasureC!(Cvals::Vector{Int}, config::Vector{Int})
    @inbounds for i in 1:length(config)
        sz_abs = abs(config[i])
        if sz_abs == 2
            Cvals[i] = 1
        elseif sz_abs == 1
            Cvals[i] = -1
        else
            Cvals[i] = 0
        end
    end
    return Cvals
end
function update!(sys::BracketSystem)
    L = sys.L
    S = sys.S
    for _ in 1:L
        l = rand(1:L)
        r = l % L + 1
        lc_old, rc_old = sys.BracketConfig[l], sys.BracketConfig[r]
        
        if lc_old == 0 && rc_old == 0
            k = rand(1:S)
            sys.BracketConfig[l], sys.BracketConfig[r] = k, -k  # 0 0 -> uᵏ dᵏ
        elseif lc_old == -rc_old && lc_old > 0
            if rand() < 1/S
                sys.BracketConfig[l], sys.BracketConfig[r] = 0, 0 # uᵏ dᵏ -> 0 0
            end
        elseif lc_old == 0 && rc_old != 0   # 0 m -> m 0
            sys.BracketConfig[l], sys.BracketConfig[r] = rc_old, 0
        elseif lc_old != 0 && rc_old == 0   # m 0 -> 0 m
            sys.BracketConfig[r], sys.BracketConfig[l] = lc_old, 0
        end
    end
end
# The main loop is now generic. It orchestrates the simulation and delegates
# measurement tasks to the observable objects.
"""
Performs updates and measurements for the generalized Spin-S model.
"""
function RunAndMeasure!(sys::BracketSystem, interval::Int, observables::Vector{<:AbstractObservable}, tmea::Vector{Int})
    L, S = sys.L, sys.S
    total_simulation_time = sys.Tthermal + sys.T
    
    # Initialize observables
    for obs in observables; initialize!(obs, sys, tmea); end

    total_measurements = 0
    start_time = time()

    @inbounds for t in 1:total_simulation_time
        # --- Step 1: Perform Monte Carlo update step ---
        update!(sys)

        # --- Step 2: Update shared history buffer (if needed) ---
        if sys.use_history
            buffer_idx = (t - 1) % sys.buffer_size + 1
            sys.Z_history[:, buffer_idx] .= sys.BracketConfig
        end
 
        # --- Step 3: Perform measurements after thermalization ---
        if t > sys.Tthermal
            total_measurements += 1
            # Create the smart workspace. No calculations happen here!
            workspace = MeasurementWorkspace(t, sys.BracketConfig, sys.S)
            
            # # Create the workspace with pre-calculated quantities for this step
            # workspace = MeasurementWorkspace(t, 
            #     MeasureO_SpinS!(workspace_O, sys.BracketConfig, S),
            #     height_profile!(workspace_H, sys.BracketConfig),
            # )

            # Delegate measurement to each registered observable
            # When an observable calls `workspace.H_vals`, the logic we defined above
            # will be triggered automatically.
            for obs in observables
                measure!(obs, sys, workspace)
            end
        end
        
        # --- Step 4: Progress Reporting ---
        if sys.rank == 0 && (t % (total_simulation_time ÷ 100) == 0 || t == total_simulation_time)
             elapsed = round(time() - start_time, digits=2)
             phase = t <= sys.Tthermal ? "Thermalizing" : "Measuring"
             progress = t <= sys.Tthermal ? t/sys.Tthermal : (t-sys.Tthermal)/sys.T
             println("$phase... $(round(100*progress, digits=1))% complete. Elapsed: $(elapsed)s.")
        end
    end

    # --- Step 5: Finalize all calculations ---
    for obs in observables
        finalize!(obs, total_measurements)
    end
end