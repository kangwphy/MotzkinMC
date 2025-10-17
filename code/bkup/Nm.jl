# File: Observables/Projectors.jl
# Contains all observables related to projectors, like Pm (|Sz|=m) and Nm (Sz=m).

using ..Observables

#==============================================================================
# Part 1: Pm Observables (|Sz| = m)
==============================================================================#

#--------------------------------------------------------------------
# Observable: General Pm Projector Expectation Value <|Sz|=m>
#--------------------------------------------------------------------
mutable struct PmExpect <: AbstractObservable
    m::Int
    sum::Float64
    final_value::Float64
    function PmExpect(m::Int)
        if m < 0
            error("Spin magnitude 'm' for PmExpect must be non-negative.")
        end
        new(m, 0.0, 0.0)
    end
end

get_filename(obs::PmExpect) = "pm$(obs.m)_expect.dat"
get_header(obs::PmExpect) = "m\tPm_Avg\tPm_Err"
get_data(obs::PmExpect) = [obs.final_value]
get_xaxis(obs::PmExpect, L::Int, tmea::Vector{Int}) = [obs.m]

function measure!(obs::PmExpect, sys, workspace)
    p_m = abs.(sys.BracketConfig) .== obs.m
    obs.sum += sum(p_m) / sys.L
end

function finalize!(obs::PmExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: General Pm Projector TCorrelation
#--------------------------------------------------------------------
mutable struct PmTCorr <: AbstractObservable
    m::Int
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function PmTCorr(tmea::Vector{Int}, m::Int)
        if m < 0
            error("Spin magnitude 'm' for PmTCorr must be non-negative.")
        end
        num_lags = length(tmea)
        new(m, tmea, zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::PmTCorr) = "pm$(obs.m)_tcorr.dat"
get_header(obs::PmTCorr) = "t\tC_Pm$(obs.m)_Avg\tC_Pm$(obs.m)_Err"
get_data(obs::PmTCorr) = obs.final_corr
get_xaxis(obs::PmTCorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::PmTCorr, sys, workspace)
    if !sys.use_history; error("PmTCorr requires history."); end
    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)
            p_m_t = abs.(current_Z) .== obs.m
            p_m_0 = abs.(past_Z) .== obs.m
            obs.prod_sum[idx] += dot(p_m_t, p_m_0) / sys.L
            obs.mean_t_sum[idx] += sum(p_m_t) / sys.L
            obs.mean_0_sum[idx] += sum(p_m_0) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::PmTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

#==============================================================================
# Part 2: Nm Observables (Sz = m)
==============================================================================#

#--------------------------------------------------------------------
# Observable: General Nm Projector Expectation Value <Sz=m>
#--------------------------------------------------------------------
mutable struct NmExpect <: AbstractObservable
    m::Int
    sum::Float64
    final_value::Float64
    function NmExpect(m::Int)
        new(m, 0.0, 0.0)
    end
end

get_filename(obs::NmExpect) = "nm$(obs.m)_expect.dat"
get_header(obs::NmExpect) = "m\tNm_Avg\tNm_Err"
get_data(obs::NmExpect) = [obs.final_value]
get_xaxis(obs::NmExpect, L::Int, tmea::Vector{Int}) = [obs.m]

function measure!(obs::NmExpect, sys, workspace)
    n_m = (sys.BracketConfig) .== obs.m
    obs.sum += sum(n_m) / sys.L
end

function finalize!(obs::NmExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: General Nm Projector TCorrelation
#--------------------------------------------------------------------
mutable struct NmTCorr <: AbstractObservable
    m1::Int
    m2::Int
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function NmTCorr(tmea::Vector{Int}, m::Int)
        num_lags = length(tmea)
        new(m, m, tmea, zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
    function NmTCorr(tmea::Vector{Int}, m1::Int, m2::Int)
        num_lags = length(tmea)
        new(m1, m2, tmea, zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::NmTCorr) = "nm$(obs.m1)m$(obs.m2)_tcorr.dat"
get_header(obs::NmTCorr) = "t\tC_Nm$(obs.m1)m$(obs.m2)_Avg\tC_Nm$(obs.m1)m$(obs.m2)_Err"
get_data(obs::NmTCorr) = obs.final_corr
get_xaxis(obs::NmTCorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::NmTCorr, sys, workspace)
    if !sys.use_history; error("NmTCorr requires history."); end
    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)
            n_m_t = (current_Z) .== obs.m1
            n_m_0 = (past_Z) .== obs.m2
            obs.prod_sum[idx] += dot(n_m_t, n_m_0) / sys.L
            obs.mean_t_sum[idx] += sum(n_m_t) / sys.L
            obs.mean_0_sum[idx] += sum(n_m_0) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::NmTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end


#==============================================================================
# Part 3: Self-registration with keyword arguments
==============================================================================#

# Register Pm observables
register_observable!("pm_expect", (L, tmea; kwargs...) -> PmExpect(kwargs[:m]))
register_observable!("pm_tcorr", (L, tmea; kwargs...) -> PmTCorr(tmea, kwargs[:m]))

# Register Nm observables
register_observable!("nm_expect", (L, tmea; kwargs...) -> NmExpect(kwargs[:m]))

# This registration handles both nm<m>_tcorr and nm<m1>m<m2>_tcorr
register_observable!("nm_tcorr", (L, tmea; kwargs...) -> begin
    if haskey(kwargs, :m1) && haskey(kwargs, :m2)
        return NmTCorr(tmea, kwargs[:m1], kwargs[:m2])
    elseif haskey(kwargs, :m)
        return NmTCorr(tmea, kwargs[:m])
    else
        error("nm_tcorr requires keyword argument 'm' or ('m1', 'm2')")
    end
end)