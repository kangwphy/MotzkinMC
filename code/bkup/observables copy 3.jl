using LinearAlgebra
using Statistics

#==============================================================================
# SECTION 1: Abstract Observable Interface
==============================================================================#

abstract type AbstractObservable end

function initialize!(obs::AbstractObservable, sys, tmea::Vector{Int})
    # Default is to do nothing
end

function measure!(obs::AbstractObservable, sys, workspace)
    error("measure! not implemented for $(typeof(obs))")
end

function finalize!(obs::AbstractObservable, total_measurements::Int)
    # Default is to do nothing
end

function get_data(obs::AbstractObservable)::AbstractVector{Float64}
    error("get_data not implemented for $(typeof(obs))")
end

function get_xaxis(obs::AbstractObservable, L::Int, tmea::Vector{Int})::AbstractVector
    error("get_xaxis not implemented for $(typeof(obs))")
end

function get_filename(obs::AbstractObservable)::String
    error("get_filename not implemented for $(typeof(obs))")
end

function get_header(obs::AbstractObservable)::String
    error("get_header not implemented for $(typeof(obs))")
end

#==============================================================================
# SECTION 1.5: Generic Helper Functions for Observables
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
# SECTION 2: Concrete Observable Implementations
==============================================================================#

#--------------------------------------------------------------------
# Observable: Average Sz Expectation Value per site <Sz(i)>
#--------------------------------------------------------------------
mutable struct SzExpect <: AbstractObservable
    mean_sum::Vector{Float64}
    final_value::Vector{Float64}
    SzExpect(L::Int) = new(zeros(Float64, L), zeros(Float64, L))
end

get_filename(obs::SzExpect) = "sz_expect.dat"
get_header(obs::SzExpect) = "Site\tSz_Avg\tSz_Err"
get_data(obs::SzExpect) = obs.final_value
get_xaxis(obs::SzExpect, L::Int, tmea::Vector{Int}) = 1:L

function measure!(obs::SzExpect, sys, workspace)
    obs.mean_sum .+= sys.BracketConfig
end

function finalize!(obs::SzExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value .= obs.mean_sum ./ total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: Equal-Time Sz-Sz Spatial Correlation
#--------------------------------------------------------------------
mutable struct SzSzRCorr <: AbstractObservable
    corr_sum::Vector{Float64}
    mean_sum::Vector{Float64}
    final_corr::Vector{Float64}
    
    function SzSzRCorr(L::Int)
        max_dist = L ÷ 2
        new(zeros(Float64, max_dist), zeros(Float64, L), zeros(Float64, max_dist))
    end
end

get_filename(obs::SzSzRCorr) = "szsz_r_corr.dat"
get_header(obs::SzSzRCorr) = "r\tSzSz_r_Avg\tSzSz_r_Err"
get_data(obs::SzSzRCorr) = obs.final_corr
get_xaxis(obs::SzSzRCorr, L::Int, tmea::Vector{Int}) = 1:(L÷2)

function measure!(obs::SzSzRCorr, sys, workspace)
    accumulate_spatial_correlation!(obs, sys.BracketConfig, sys.L)
end

function finalize!(obs::SzSzRCorr, total_measurements::Int)
    finalize_spatial_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# Observable: Sz-Sz Time Correlation
#--------------------------------------------------------------------
mutable struct SzSzTCorr <: AbstractObservable
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function SzSzTCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        new(tmea, zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::SzSzTCorr) = "szsz_t_corr.dat"
get_header(obs::SzSzTCorr) = "t\tSzSz_t_Avg\tSzSz_t_Err"
get_data(obs::SzSzTCorr) = obs.final_corr
get_xaxis(obs::SzSzTCorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::SzSzTCorr, sys, workspace)
    if !sys.use_history
        error("SzSzTCorr requires the system's history buffer to be enabled.")
    end

    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)

    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)

            obs.prod_sum[idx] += dot(current_Z, past_Z) / sys.L
            obs.mean_t_sum[idx] += sum(current_Z) / sys.L
            obs.mean_0_sum[idx] += sum(past_Z) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::SzSzTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# Observable: O Operator Expectation Value <O>
#--------------------------------------------------------------------
mutable struct OExpect <: AbstractObservable
    sum::Float64
    final_value::Float64
    OExpect() = new(0.0, 0.0)
end

get_filename(obs::OExpect) = "o_expect.dat"
get_header(obs::OExpect) = "Observable\tAvg\tErr"
get_data(obs::OExpect) = [obs.final_value]
get_xaxis(obs::OExpect, L::Int, tmea::Vector{Int}) = ["<O>"]

function measure!(obs::OExpect, sys, workspace)
    obs.sum += sum(workspace.O_vals) / sys.L
end

function finalize!(obs::OExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: Equal-Time O-O Spatial Correlation
#--------------------------------------------------------------------
mutable struct OORCorr <: AbstractObservable
    corr_sum::Vector{Float64}
    mean_sum::Vector{Float64}
    final_corr::Vector{Float64}
    function OORCorr(L::Int)
        max_dist = L ÷ 2
        new(zeros(Float64, max_dist), zeros(Float64, L), zeros(Float64, max_dist))
    end
end

get_filename(obs::OORCorr) = "oo_r_corr.dat"
get_header(obs::OORCorr) = "r\tOO_r_Avg\tOO_r_Err"
get_data(obs::OORCorr) = obs.final_corr
get_xaxis(obs::OORCorr, L::Int, tmea::Vector{Int}) = 1:(L÷2)

function measure!(obs::OORCorr, sys, workspace)
    accumulate_spatial_correlation!(obs, workspace.O_vals, sys.L)
end

function finalize!(obs::OORCorr, total_measurements::Int)
    finalize_spatial_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# Observable: O-O Time Correlation
#--------------------------------------------------------------------
mutable struct OOTCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    O_history::Matrix{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function OOTCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, buffer_size, Matrix{Int}(undef, 0, 0),
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::OOTCorr) = "oo_t_corr.dat"
get_header(obs::OOTCorr) = "t\tOO_t_Avg\tOO_t_Err"
get_data(obs::OOTCorr) = obs.final_corr
get_xaxis(obs::OOTCorr, L::Int, tmea::Vector{Int}) = tmea

# This observable needs to allocate its own history buffer
function initialize!(obs::OOTCorr, sys, tmea::Vector{Int})
    obs.O_history = zeros(Int, sys.L, obs.buffer_size)
end

function measure!(obs::OOTCorr, sys, workspace)
    current_O = workspace.O_vals
    current_buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
    obs.O_history[:, current_buffer_idx] .= current_O
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % obs.buffer_size + 1
            past_O = view(obs.O_history, :, past_buffer_idx)
            obs.prod_sum[idx] += dot(current_O, past_O) / sys.L
            obs.mean_t_sum[idx] += sum(current_O) / sys.L
            obs.mean_0_sum[idx] += sum(past_O) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::OOTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# Observable: Average Height Expectation Value per site <h(i)>
#--------------------------------------------------------------------
mutable struct HeightExpect <: AbstractObservable
    mean_sum::Vector{Float64}
    final_value::Vector{Float64}
    HeightExpect(L::Int) = new(zeros(Float64, L), zeros(Float64, L))
end

get_filename(obs::HeightExpect) = "height_expect.dat"
get_header(obs::HeightExpect) = "Site\th_Avg\th_Err"
get_data(obs::HeightExpect) = obs.final_value
get_xaxis(obs::HeightExpect, L::Int, tmea::Vector{Int}) = 1:L

function measure!(obs::HeightExpect, sys, workspace)
    # H_vals (the height profile) is already calculated in the workspace
    obs.mean_sum .+= workspace.H_vals
end

function finalize!(obs::HeightExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value .= obs.mean_sum ./ total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: Height-Height Time Correlation at the center point L/2
# C(t) = <h(L/2,t)h(L/2,0)> - <h(L/2,t)><h(L/2,0)>
#--------------------------------------------------------------------
mutable struct HeightTimeCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    H_mid_history::Vector{Float64} # Local history for the midpoint height
    
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    
    function HeightTimeCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, buffer_size, zeros(Float64, buffer_size),
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::HeightTimeCorr) = "height_tcorr.dat"
get_header(obs::HeightTimeCorr) = "t\tHH_t_Avg\tHH_t_Err"
get_data(obs::HeightTimeCorr) = obs.final_corr
get_xaxis(obs::HeightTimeCorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::HeightTimeCorr, sys, workspace)
    current_H_mid = workspace.H_vals[sys.L ÷ 2]
    
    # Store current midpoint height in the local history buffer
    buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
    obs.H_mid_history[buffer_idx] = current_H_mid

    # Correlate with past values
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_idx = (workspace.t - lag - 1) % obs.buffer_size + 1
            past_H_mid = obs.H_mid_history[past_idx]
            
            # Accumulate sums for the connected correlator
            obs.prod_sum[idx] += current_H_mid * past_H_mid
            obs.mean_t_sum[idx] += current_H_mid
            obs.mean_0_sum[idx] += past_H_mid
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::HeightTimeCorr, total_measurements::Int)
    # The calculation logic is identical to other TCorr observables
    finalize_time_correlation!(obs, total_measurements)
end

#--------------------------------------------------------------------
# Observable: Height Diffusion A(t) 
#--------------------------------------------------------------------
mutable struct HeightDiffusion <: AbstractObservable
    tmea::Vector{Int}
    H_mid_history::Vector{Float64}
    buffer_size::Int
    h_sq_diff_sum::Vector{Float64}
    h_0_sq_sum::Vector{Float64}
    h_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_A::Vector{Float64}
    function HeightDiffusion(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, zeros(Float64, buffer_size), buffer_size,
            zeros(Float64, num_lags), zeros(Float64, num_lags), 
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::HeightDiffusion) = "height_diffusion.dat"
get_header(obs::HeightDiffusion) = "t\tHeight_Diff_A_Avg\tHeight_Diff_A_Err"
get_data(obs::HeightDiffusion) = obs.final_A
get_xaxis(obs::HeightDiffusion, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::HeightDiffusion, sys, workspace)
    current_H_mid = workspace.H_vals[sys.L ÷ 2]
    buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
    obs.H_mid_history[buffer_idx] = current_H_mid
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_idx = (workspace.t - lag - 1) % obs.buffer_size + 1
            past_H_mid = obs.H_mid_history[past_idx]
            obs.h__diff_sum[idx] += (current_H_mid - past_H_mid)^2
            obs.h_0_sq_sum[idx] += past_H_mid^2
            obs.h_0_sum[idx] += past_H_mid
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::HeightDiffusion, total_measurements::Int)
    for idx in 1:length(obs.tmea)
        num_meas = obs.num_lag_measurements[idx]
        if num_meas > 0
            avg_h_sq_diff = obs.h_sq_diff_sum[idx] / num_meas
            avg_h_0_sq = obs.h_0_sq_sum[idx] / num_meas
            avg_h_0 = obs.h_0_sum[idx] / num_meas
            denom = avg_h_0_sq - avg_h_0^2
            obs.final_A[idx] = (denom > 1e-9) ? 2.0 - avg_h_sq_diff / denom : 0.0
        end
    end
end

#--------------------------------------------------------------------
# Observable: Spin Magnitude Autocorrelation - (Specific name, no change)
#--------------------------------------------------------------------
mutable struct SpinMagnitudeAutocorr <: AbstractObservable
    tmea::Vector{Int}
    match_count::Vector{Int}
    nonzero_initial_count::Vector{Int}
    final_corr::Vector{Float64}
    function SpinMagnitudeAutocorr(tmea::Vector{Int})
        num_lags = length(tmea)
        new(tmea, zeros(Int, num_lags), zeros(Int, num_lags), zeros(Float64, num_lags))
    end
end

get_filename(obs::SpinMagnitudeAutocorr) = "spin_mag_autocorr.dat"
get_header(obs::SpinMagnitudeAutocorr) = "t\tP_Avg\tP_Err"
get_data(obs::SpinMagnitudeAutocorr) = obs.final_corr
get_xaxis(obs::SpinMagnitudeAutocorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::SpinMagnitudeAutocorr, sys, workspace)
    if !sys.use_history
        error("SpinMagnitudeAutocorr requires the system's history buffer to be enabled.")
    end
    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)
            for i in 1:sys.L
                sz_0 = past_Z[i]
                if sz_0 != 0
                    obs.nonzero_initial_count[idx] += 1
                    sz_t = current_Z[i]
                    if abs(sz_t) == abs(sz_0)
                        obs.match_count[idx] += 1
                    end
                end
            end
        end
    end
end

function finalize!(obs::SpinMagnitudeAutocorr, total_measurements::Int)
    for idx in 1:length(obs.tmea)
        if obs.nonzero_initial_count[idx] > 0
            obs.final_corr[idx] = obs.match_count[idx] / obs.nonzero_initial_count[idx]
        else
            obs.final_corr[idx] = 0.0
        end
    end
end

#--------------------------------------------------------------------
# Observable: General Pm Projector Expectation Value <|Sz|=m>
#--------------------------------------------------------------------
mutable struct PmExpect <: AbstractObservable
    m::Int
    sum::Float64
    final_value::Float64
    function PmExpect(m::Int)
        if m < 0
            error("Spin magnitude 'm' for PmExpect must be a positive integer.")
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
# Observable: General Pm Projector Autocorrelation - (Specific, no change)
#--------------------------------------------------------------------
mutable struct PmAutocorr <: AbstractObservable
    m::Int
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function PmAutocorr(tmea::Vector{Int}, m::Int)
        if m < 0
            error("Spin magnitude 'm' for PmAutocorr must be a positive integer.")
        end
        num_lags = length(tmea)
        new(m, tmea, zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::PmAutocorr) = "pm$(obs.m)_autocorr.dat"
get_header(obs::PmAutocorr) = "t\tC_Pm$(obs.m)_Avg\tC_Pm$(obs.m)_Err"
get_data(obs::PmAutocorr) = obs.final_corr
get_xaxis(obs::PmAutocorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::PmAutocorr, sys, workspace)
    if !sys.use_history
        error("PmAutocorr requires the system's history buffer to be enabled.")
    end
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

function finalize!(obs::PmAutocorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end

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
# Observable: General Nm Projector Autocorrelation - (Specific, no change)
#--------------------------------------------------------------------
mutable struct NmAutocorr <: AbstractObservable
    m1::Int
    m2::Int
    tmea::Vector{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function NmAutocorr(tmea::Vector{Int}, m::Int)
        num_lags = length(tmea)
        new(m, m, tmea, zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
    function NmAutocorr(tmea::Vector{Int}, m1::Int,m2::Int)
        num_lags = length(tmea)
        new(m1, m2, tmea, zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::NmAutocorr) = "nm$(obs.m1)m$(obs.m2)_autocorr.dat"
get_header(obs::NmAutocorr) = "t\tC_Nm$(obs.m1)m$(obs.m2)_Avg\tC_Nm$(obs.m1)m$(obs.m2)_Err"
get_data(obs::NmAutocorr) = obs.final_corr
get_xaxis(obs::NmAutocorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::NmAutocorr, sys, workspace)
    if !sys.use_history
        error("NmAutocorr requires the system's history buffer to be enabled.")
    end
    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)
            p_m_t = (current_Z) .== obs.m1
            p_m_0 = (past_Z) .== obs.m2
            obs.prod_sum[idx] += dot(p_m_t, p_m_0) / sys.L
            obs.mean_t_sum[idx] += sum(p_m_t) / sys.L
            obs.mean_0_sum[idx] += sum(p_m_0) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::NmAutocorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end
#--------------------------------------------------------------------
# Observable: C Operator Expectation Value <C>
#--------------------------------------------------------------------
mutable struct CExpect <: AbstractObservable
    sum::Float64
    final_value::Float64
    CExpect() = new(0.0, 0.0)
end

get_filename(obs::CExpect) = "c_expect.dat"
get_header(obs::CExpect) = "Observable\tAvg\tErr"
get_data(obs::CExpect) = [obs.final_value]
get_xaxis(obs::CExpect, L::Int, tmea::Vector{Int}) = ["<C>"]

function measure!(obs::CExpect, sys, workspace)
    obs.sum += sum(workspace.C_vals) / sys.L
end

function finalize!(obs::CExpect, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: C-C Time Correlation
#--------------------------------------------------------------------
mutable struct CCTCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    C_history::Matrix{Int}
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}
    function CCTCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        new(tmea, buffer_size, Matrix{Int}(undef, 0, 0),
            zeros(Float64, num_lags), zeros(Float64, num_lags),
            zeros(Float64, num_lags), zeros(Int, num_lags),
            zeros(Float64, num_lags))
    end
end

get_filename(obs::CCTCorr) = "cc_t_corr.dat"
get_header(obs::CCTCorr) = "t\tCC_t_Avg\tCC_t_Err"
get_data(obs::CCTCorr) = obs.final_corr
get_xaxis(obs::CCTCorr, L::Int, tmea::Vector{Int}) = tmea

function initialize!(obs::CCTCorr, sys, tmea::Vector{Int})
    obs.C_history = zeros(Int, sys.L, obs.buffer_size)
end

function measure!(obs::CCTCorr, sys, workspace)
    current_C = workspace.C_vals
    current_buffer_idx = (workspace.t - 1) % obs.buffer_size + 1
    obs.C_history[:, current_buffer_idx] .= current_C
    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % obs.buffer_size + 1
            past_C = view(obs.C_history, :, past_buffer_idx)
            obs.prod_sum[idx] += dot(current_C, past_C) / sys.L
            obs.mean_t_sum[idx] += sum(current_C) / sys.L
            obs.mean_0_sum[idx] += sum(past_C) / sys.L
            obs.num_lag_measurements[idx] += 1
        end
    end
end

function finalize!(obs::CCTCorr, total_measurements::Int)
    finalize_time_correlation!(obs, total_measurements)
end
