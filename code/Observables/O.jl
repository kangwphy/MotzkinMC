# File: Observables/O.jl
# Contains all observables related to the O operator, now using the generic framework.

using ..Observables

# 1. Define the calculation logic for the O operator.
#    This function used to be in sub.jl.
const O_KEY = :O_vals

function measure_o!(Ovals::Vector{Int}, config::Vector{Int}, S::Int)
    L = length(config)
    @inbounds for i in 1:L
        j = (i % L) + 1
        l, r = config[i], config[j]
        Ovals[i] = (l == 0 && r == 0) ? S : ((l < 0 && l == -r) ? 1 : 0)
    end
    return Ovals
end

# 2. Register this calculation logic as a "derived quantity".
register_derived_quantity!(O_KEY, (config, S) -> begin
    Ovals = zeros(Int, length(config))
    measure_o!(Ovals, config, S)
    return Ovals
end)

# 3. Register observables that USE this quantity.
register_observable!("o_expect", (L, tmea; kwargs...) -> begin
    filename = "o_expect.dat"
    header = "Observable\tO_Avg\tO_Err"
    xaxis = ["O"]
    return GeneralExpect(filename, header, xaxis, O_KEY)
end)

register_observable!("oo_rcorr", (L, tmea; kwargs...) -> begin
    filename = "oo_rcorr.dat"
    header = "r\tOO_r_Avg\tOO_r_Err"
    return GeneralRCorr(filename, header, O_KEY, L)
end)

register_observable!("oo_tcorr", (L, tmea; kwargs...) -> begin
    filename = "oo_tcorr.dat"
    header = "t\tOO_t_Avg\tOO_t_Err"
    return GeneralTCorr(filename, header, tmea, O_KEY, L, Int)
end)


#==============================================================================
# 1. Calculation Logic and Registration
==============================================================================#
# --- Nm Operator ---
const OMN_KEY = :Omn_vals

"""
    measure_omn!(Ovals::Vector{Int}, config::Vector{Int}, S::Int, m::Int, n::Int)

Updates `Ovals` based on an ordered sliding window analysis of `config`.

For each position `i` in `config`, it inspects a segment of length `m + n` starting at `i` (with wrap-around).

- If all `m + n` elements in the segment are `0`, `Ovals[i]` is set to `S`.
- If the first `m` elements of the segment are all equal to `-k` and the following `n` elements are all equal to `k` (for any integer `k > 0`), `Ovals[i]` is set to `1`.
- Otherwise, `Ovals[i]` is set to `0`.
"""
function measure_omn!(Ovals::Vector{Int}, config::Vector{Int}, S::Int, m::Int, n::Int)
    L = length(config)
    segl = m + n

    @inbounds for i in 1:L
        # --- Check for the 'S' condition: a segment of all zeros ---
        # This logic remains the same as it is correct.
        is_all_zeros = true
        for j in 0:segl-1
            idx = mod1(i + j, L)
            if config[idx] != 0
                is_all_zeros = false
                break
            end
        end

        if is_all_zeros
            Ovals[i] = S
            continue # Move to the next `i`
        end

        # --- Check for the '1' condition: m negative-k followed by n positive-k ---
        # This logic is now updated to enforce strict ordering.

        # Determine the expected value `k` from the first element of the pattern.
        # If m > 0, the first element must be negative.
        # If m = 0 (and n > 0), the first element must be positive.
        k_val = 0
        first_idx = mod1(i, L)
        if m > 0
            first_val = config[first_idx]
            if first_val >= 0
                # Pattern fails: the first element of the m-block must be negative.
                Ovals[i] = 0
                continue
            end
            k_val = -first_val # This is abs(first_val)
        elseif n > 0 # Case where m=0
             first_val = config[first_idx]
             if first_val <= 0
                # Pattern fails: the first element of the n-block must be positive.
                Ovals[i] = 0
                continue
             end
             k_val = first_val
        end

        # Now, verify the entire ordered pattern.
        is_ordered_pattern = true
        # Check the first `m` elements
        for j in 0:m-1
            idx = mod1(i + j, L)
            if config[idx] != -k_val
                is_ordered_pattern = false
                break
            end
        end

        # If the m-block was valid, check the next `n` elements
        if is_ordered_pattern
            for j in m:segl-1
                idx = mod1(i + j, L)
                if config[idx] != k_val
                    is_ordered_pattern = false
                    break
                end
            end
        end

        if is_ordered_pattern
            Ovals[i] = 1
        else
            Ovals[i] = 0
        end
    end

    return Ovals
end

# 2. Register this calculation logic as a "derived quantity".
register_derived_quantity!(OMN_KEY, (config, S; m, n) -> begin
    Ovals = zeros(Int, length(config))
    measure_omn!(Ovals, config, S, m, n)
    return Ovals
end)


#==============================================================================
# 2. Observable Registration
==============================================================================#

# --- Register Omn observables ---
register_observable!("omn_expect", (L, tmea; m, n, kwargs...) -> begin
    key_kwargs = (m=m, n=n)
    filename = "om$(m)n$(n)_expect.dat"
    header = "mn\tOmn_Avg\tOmn_Err"
    xaxis = ["Om$(m)n$(n)"]
    return GeneralExpect(filename, header, xaxis, OMN_KEY; key_kwargs = key_kwargs)
end)

register_observable!("omn_tcorr", (L, tmea; m=nothing, n=nothing, kwargs...) -> begin
    kwargs_t = (m=m, n=n,)
    kwargs_0 = (m=m, n=n,)
    
    filename = "om$(m)n$(n)m$(m)n$(n)_tcorr.dat"
    header = "t\tOm$(m)n$(n)m$(m)n$(n)_Avg\tOm$(m)n$(n)m$(m)n$(n)_Err"
    
    # Call the main constructor for cross-correlation
    return GeneralTCorr(filename, header, tmea, OMN_KEY, OMN_KEY, L, Int; kwargs_t = kwargs_t, kwargs_0 = kwargs_0)
end)

################### Q operators ###################
# File: Observables/O.jl
# Contains all observables related to the O operator, now using the generic framework.

using ..Observables

# 1. Define the calculation logic for the O operator.
#    This function used to be in sub.jl.
const Q_KEY = :Q_vals

function measure_q!(Qvals::Vector{Int}, config::Vector{Int}, S::Int)
    L = length(config)
    @inbounds for i in 1:L
        j = (i % L) + 1
        l, r = config[i], config[j]
        Qvals[i] = (l == 0 && r == 0) ? S : ((l > 0 && l == -r) ? 1 : 0)
    end
    return Qvals
end

# 2. Register this calculation logic as a "derived quantity".
register_derived_quantity!(Q_KEY, (config, S) -> begin
    Qvals = zeros(Int, length(config))
    measure_q!(Qvals, config, S)
    return Qvals
end)

# 3. Register observables that USE this quantity.
register_observable!("q_expect", (L, tmea; kwargs...) -> begin
    filename = "q_expect.dat"
    header = "Observable\tQ_Avg\tQ_Err"
    xaxis = ["Q"]
    return GeneralExpect(filename, header, xaxis, Q_KEY)
end)

register_observable!("qq_rcorr", (L, tmea; kwargs...) -> begin
    filename = "qq_rcorr.dat"
    header = "r\tQQ_r_Avg\tQQ_r_Err"
    return GeneralRCorr(filename, header, Q_KEY, L)
end)

register_observable!("qq_tcorr", (L, tmea; kwargs...) -> begin
    filename = "qq_tcorr.dat"
    header = "t\tQQ_t_Avg\tQQ_t_Err"
    return GeneralTCorr(filename, header, tmea, Q_KEY, L, Int)
end)



#==============================================================================
# 1. Calculation Logic and Registration
==============================================================================#
# --- Nm Operator ---
const QMN_KEY = :Qmn_vals

"""
    measure_qmn!(Qvals::Vector{Int}, config::Vector{Int}, S::Int, m::Int, n::Int)

Updates `Qvals` based on an ordered sliding window analysis of `config`.

This is the reverse version of `measure_omn!`. For each position `i` in `config`,
it inspects a segment of length `m + n` starting at `i` (with wrap-around).

- If all `m + n` elements in the segment are `0`, `Qvals[i]` is set to `S`.
- If the first `m` elements of the segment are all equal to `k` and the following `n` elements are all equal to `-k` (for any integer `k > 0`), `Qvals[i]` is set to `1`.
- Otherwise, `Qvals[i]` is set to `0`.
"""
function measure_qmn!(Qvals::Vector{Int}, config::Vector{Int}, S::Int, m::Int, n::Int)
    L = length(config)
    segl = m + n

    # If the segment length is zero, no condition can be met.
    if segl == 0
        fill!(Qvals, 0)
        return Qvals
    end

    @inbounds for i in 1:L
        # --- Check for the 'S' condition: a segment of all zeros ---
        is_all_zeros = true
        for j in 0:segl-1
            idx = mod1(i + j, L)
            if config[idx] != 0
                is_all_zeros = false
                break
            end
        end

        if is_all_zeros
            Qvals[i] = S
            continue # Move to the next `i`
        end

        # --- Check for the '1' condition: m positive-k followed by n negative-k ---
        # This is the reversed logic from measure_omn!

        # Determine the expected value `k` from the first element of the pattern.
        # If m > 0, the first element must be positive.
        # If m = 0 (and n > 0), the first element must be negative.
        k_val = 0
        first_idx = mod1(i, L)
        if m > 0
            first_val = config[first_idx]
            if first_val <= 0
                # Pattern fails: the first element of the m-block must be positive.
                Qvals[i] = 0
                continue
            end
            k_val = first_val
        elseif n > 0 # Case where m=0
             first_val = config[first_idx]
             if first_val >= 0
                # Pattern fails: the first element of the n-block must be negative.
                Qvals[i] = 0
                continue
             end
             k_val = -first_val # This is abs(first_val)
        end

        # Now, verify the entire ordered pattern.
        is_ordered_pattern = true
        # Check the first `m` elements
        for j in 0:m-1
            idx = mod1(i + j, L)
            if config[idx] != k_val
                is_ordered_pattern = false
                break
            end
        end

        # If the m-block was valid, check the next `n` elements
        if is_ordered_pattern
            for j in m:segl-1
                idx = mod1(i + j, L)
                if config[idx] != -k_val
                    is_ordered_pattern = false
                    break
                end
            end
        end

        if is_ordered_pattern
            Qvals[i] = 1
        else
            Qvals[i] = 0
        end
    end

    return Qvals
end

# 2. Register this calculation logic as a "derived quantity".
register_derived_quantity!(QMN_KEY, (config, S; m, n) -> begin
    Qvals = zeros(Int, length(config))
    measure_qmn!(Qvals, config, S, m, n)
    return Qvals
end)


#==============================================================================
# 2. Observable Registration
==============================================================================#

# --- Register Qmn observables ---
register_observable!("qmn_expect", (L, tmea; m, n, kwargs...) -> begin
    key_kwargs = (m=m, n=n)
    filename = "qm$(m)n$(n)_expect.dat"
    header = "mn\tQmn_Avg\tQmn_Err"
    xaxis = ["Qm$(m)n$(n)"]
    return GeneralExpect(filename, header, xaxis, QMN_KEY; key_kwargs = key_kwargs)
end)

register_observable!("qmn_tcorr", (L, tmea; m=nothing, n=nothing, kwargs...) -> begin
    kwargs_t = (m=m, n=n,)
    kwargs_0 = (m=m, n=n,)
    
    filename = "qm$(m)n$(n)m$(m)n$(n)_tcorr.dat"
    header = "t\tQm$(m)n$(n)m$(m)n$(n)_Avg\tQm$(m)n$(n)m$(m)n$(n)_Err"
    
    # Call the main constructor for cross-correlation
    return GeneralTCorr(filename, header, tmea, QMN_KEY, QMN_KEY, L, Int; kwargs_t = kwargs_t, kwargs_0 = kwargs_0)
end)

################### Q operators ###################


################### Cross Correlation ###################

register_observable!("omnqmn_tcorr", (L, tmea; m=nothing, n=nothing, kwargs...) -> begin
    kwargs_t = (m=m, n=n,)
    kwargs_0 = (m=m, n=n,)
    
    filename = "om$(m)n$(n)qm$(m)n$(n)_tcorr.dat"
    header = "t\tOm$(m)n$(n)m$(m)n$(n)_Avg\tOm$(m)n$(n)Qm$(m)n$(n)_Err"
    
    # Call the main constructor for cross-correlation
    return GeneralTCorr(filename, header, tmea, OMN_KEY, QMN_KEY, L, Int; kwargs_t = kwargs_t, kwargs_0 = kwargs_0)
end)