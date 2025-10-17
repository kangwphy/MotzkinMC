# File: Observables/Projectors.jl
# This file now acts as a configuration, registering observables
# that are implemented by the generic structs in Generic.jl.

using ..Observables

#==============================================================================
# Registration for Projector Observables
==============================================================================#

# Register Pm observables
register_observable!("pm_expect", (L, tmea; m, kwargs...) -> begin
    predicate = s -> abs(s) == m  # The "what to measure" logic
    filename = "pm$(m)_expect.dat"
    header = "m\tPm_Avg\tPm_Err"
    xaxis = [m]
    return GeneralExpect(filename, header, xaxis, predicate)
end)

register_observable!("pm_tcorr", (L, tmea; m=nothing, m1=nothing, m2=nothing, kwargs...) -> begin
    # predicate = s -> abs(s) == m
    final_m1 = m1 !== nothing ? m1 : m
    final_m2 = m2 !== nothing ? m2 : m
    if final_m1 === nothing || final_m2 === nothing
        error("pm_tcorr requires keyword argument 'm' or ('m1', 'm2')")
    end
    predicate_t = s -> abs(s) == final_m1
    predicate_0 = s -> abs(s) == final_m2
    filename = "pm$(m)_tcorr.dat"
    header = "t\tPm$(m)_t_Avg\tPm$(m)_t_Err"
    return GeneralTCorr(filename, header, tmea, predicate_t, predicate_0, L)
end)

# Register Nm observables
register_observable!("nm_expect", (L, tmea; m, kwargs...) -> begin
    predicate = s -> s == m
    filename = "nm$(m)_expect.dat"
    header = "m\tNm_Avg\tNm_Err"
    xaxis = [m]
    return GeneralExpect(filename, header, xaxis, predicate)
end)

register_observable!("nm_tcorr", (L, tmea; m=nothing, m1=nothing, m2=nothing, kwargs...) -> begin
    # Logic to handle both single 'm' and 'm1, m2' cases
    final_m1 = m1 !== nothing ? m1 : m
    final_m2 = m2 !== nothing ? m2 : m
    if final_m1 === nothing || final_m2 === nothing
        error("nm_tcorr requires keyword argument 'm' or ('m1', 'm2')")
    end
    
    predicate_t = s -> s == final_m1
    predicate_0 = s -> s == final_m2
    filename = "nm$(final_m1)m$(final_m2)_tcorr.dat"
    header = "t\tNm$(final_m1)m$(final_m2)_t_Avg\tNm$(final_m1)m$(final_m2)_t_Err"
    return GeneralTCorr(filename, header, tmea, predicate_t, predicate_0, L)
end)


# 1. Define the calculation logic for the C operator.
const A_KEY = :A_vals # Define a unique key for this quantity

function measure_a!(Avals::Vector{Int}, config::Vector{Int})
    @inbounds for i in 1:length(config)
        Avals[i] = (config[i] == 0 ? 1 : 0)
    end
    return Avals
end

# 2. Register this calculation logic as a "derived quantity".
register_derived_quantity!(A_KEY, (config, S) -> begin
    Avals = zeros(Int, length(config))
    measure_a!(Avals, config)
    return Avals
end)

register_observable!("a_expect", (L, tmea; kwargs...) -> begin
    filename = "a_expect.dat"
    header = "name\tA_Avg\tA_Err"
    xaxis = ["A"]
    return GeneralExpect(filename, header, xaxis, A_KEY)
end)

register_observable!("aa_tcorr", (L, tmea; kwargs...) -> begin    
    filename = "aa_tcorr.dat"
    header = "t\tAA_t_Avg\tAA_t_Err"
    return GeneralTCorr(filename, header, tmea, A_KEY, L, Int)
end)


# Register B observables
"""
define B = sign(s)
"""

# 1. Define the calculation logic for the C operator.
const B_KEY = :B_vals # Define a unique key for this quantity

function measure_b!(Bvals::Vector{Int}, config::Vector{Int})
    @inbounds for i in 1:length(config)
        Bvals[i] = sign(config[i])
    end
    return Bvals
end

# 2. Register this calculation logic as a "derived quantity".
register_derived_quantity!(B_KEY, (config, S) -> begin
    Bvals = zeros(Int, length(config))
    measure_b!(Bvals, config)
    return Bvals
end)

register_observable!("b_expect", (L, tmea; kwargs...) -> begin
    filename = "b_expect.dat"
    header = "name\tB_Avg\tB_Err"
    xaxis = ["B"]
    return GeneralExpect(filename, header, xaxis, B_KEY)
end)

register_observable!("bb_tcorr", (L, tmea; kwargs...) -> begin    
    filename = "bb_tcorr.dat"
    header = "t\tBB_t_Avg\tBB_t_Err"
    return GeneralTCorr(filename, header, tmea, B_KEY, L, Int)
end)


# Register C observables
"""
define C = P2-P1
"""

# 1. Define the calculation logic for the C operator.
const C_KEY = :C_vals # Define a unique key for this quantity

function measure_c!(Cvals::Vector{Int}, config::Vector{Int})
    @inbounds for i in 1:length(config)
        sz_abs = abs(config[i])
        if sz_abs == 2; Cvals[i] = 1
        elseif sz_abs == 1; Cvals[i] = -1
        else; Cvals[i] = 0
        end
    end
    return Cvals
end

# 2. Register this calculation logic as a "derived quantity".
register_derived_quantity!(C_KEY, (config, S) -> begin
    Cvals = zeros(Int, length(config))
    measure_c!(Cvals, config)
    return Cvals
end)

register_observable!("c_expect", (L, tmea; kwargs...) -> begin
    filename = "c_expect.dat"
    header = "name\tC_Avg\tC_Err"
    xaxis = ["C"]
    return GeneralExpect(filename, header, xaxis, C_KEY)
end)

register_observable!("cc_tcorr", (L, tmea; kwargs...) -> begin
    filename = "cc_tcorr.dat"
    header = "t\tCC_t_Avg\tCC_t_Err"
    return GeneralTCorr(filename, header, tmea, C_KEY, L, Int)
end)


# Register D observables
"""
define D = P2-P1
"""

# 1. Define the calculation logic for the C operator.
const D_KEY = :D_vals # Define a unique key for this quantity

function measure_d!(Dvals::Vector{Int}, config::Vector{Int})
    @inbounds for i in 1:length(config)
        s = abs(config[i])
        if s == 2 || s == -1
            Dvals[i] = 1
        elseif s == -2 || s == 1
            Dvals[i] = -1
        else
            Dvals[i] = 0
        end
    end
    return Dvals
end

# 2. Register this calculation logic as a "derived quantity".
register_derived_quantity!(D_KEY, (config, S) -> begin
    Dvals = zeros(Int, length(config))
    measure_d!(Dvals, config)
    return Dvals
end)

register_observable!("d_expect", (L, tmea; kwargs...) -> begin
    filename = "d_expect.dat"
    header = "name\tD_Avg\tD_Err"
    xaxis = ["D"]
    return GeneralExpect(filename, header, xaxis, D_KEY)
end)

register_observable!("dd_tcorr", (L, tmea; kwargs...) -> begin
    filename = "dd_tcorr.dat"
    header = "t\tDD_t_Avg\tDD_t_Err"
    return GeneralTCorr(filename, header, tmea, D_KEY, L, Int)
end)