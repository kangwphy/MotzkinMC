using LinearAlgebra
using Statistics

#==============================================================================
# SECTION 1: Abstract Observable Interface
#
# This defines the common interface that all observables must implement.
# The main simulation loop can treat any concrete observable uniformly.
==============================================================================#

abstract type AbstractObservable end

"""
Allocates memory or performs setup for the observable before the simulation.
"""
function initialize!(obs::AbstractObservable, sys, tmea::Vector{Int})
    # Default is to do nothing
end

"""
Performs a measurement on the current system state.
'sys' is the main system object.
'workspace' contains pre-calculated quantities for the current timestep to avoid redundant work.
"""
function measure!(obs::AbstractObservable, sys, workspace)
    error("measure! not implemented for $(typeof(obs))")
end

"""
Processes accumulated data into a final result after the simulation loop is complete.
"""
function finalize!(obs::AbstractObservable, total_measurements::Int)
    # Default is to do nothing
end

"""
Returns the final calculated data for MPI gathering.
"""
function get_data(obs::AbstractObservable)::AbstractVector{Float64}
    error("get_data not implemented for $(typeof(obs))")
end

"""
Returns the appropriate x-axis values for plotting (e.g., r, t).
"""
function get_xaxis(obs::AbstractObservable, L::Int, tmea::Vector{Int})::AbstractVector
    error("get_xaxis not implemented for $(typeof(obs))")
end

"""
Returns the designated output filename for the observable.
"""
function get_filename(obs::AbstractObservable)::String
    error("get_filename not implemented for $(typeof(obs))")
end

"""
Returns the header line for the output data file.
"""
function get_header(obs::AbstractObservable)::String
    error("get_header not implemented for $(typeof(obs))")
end


#==============================================================================
# SECTION 2: Concrete Observable Implementations
==============================================================================#

#--------------------------------------------------------------------
# Observable: Average Sz Profile <Sz(i)>
#--------------------------------------------------------------------
mutable struct SzProfile <: AbstractObservable
    mean_sum::Vector{Float64}
    final_profile::Vector{Float64}

    SzProfile(L::Int) = new(zeros(Float64, L), zeros(Float64, L))
end

get_filename(obs::SzProfile) = "sz_profile.dat"
get_header(obs::SzProfile) = "Site\tSz_Avg\tSz_Err"
get_data(obs::SzProfile) = obs.final_profile
get_xaxis(obs::SzProfile, L::Int, tmea::Vector{Int}) = 1:L

function measure!(obs::SzProfile, sys, workspace)
    obs.mean_sum .+= sys.BracketConfig
end

function finalize!(obs::SzProfile, total_measurements::Int)
    if total_measurements > 0
        obs.final_profile .= obs.mean_sum ./ total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: Equal-Time Sz-Sz Correlation C(r) = <Sz(i)Sz(i+r)> - <Sz(i)><Sz(i+r)>
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
get_header(obs::SzSzRCorr) = "r\tSzSz_Avg\tSzSz_Err"
get_data(obs::SzSzRCorr) = obs.final_corr
get_xaxis(obs::SzSzRCorr, L::Int, tmea::Vector{Int}) = 1:(L÷2)

function measure!(obs::SzSzRCorr, sys, workspace)
    L = sys.L
    config = sys.BracketConfig
    max_dist = L ÷ 2
    
    obs.mean_sum .+= config
    
    for r in 1:max_dist
        prod_sum = 0.0
        for i in 1:L
            j = (i + r - 1) % L + 1
            prod_sum += config[i] * config[j]
        end
        obs.corr_sum[r] += prod_sum / L
    end
end

function finalize!(obs::SzSzRCorr, total_measurements::Int)
    if total_measurements == 0 return end
    
    L = length(obs.mean_sum)
    max_dist = length(obs.corr_sum)
    
    avg_szsz_r = obs.corr_sum ./ total_measurements
    mean_sz_i = obs.mean_sum ./ total_measurements

    mean_prod_sz = zeros(Float64, max_dist)
    for r in 1:max_dist
        ps = 0.0
        for i in 1:L
            j = (i + r - 1) % L + 1
            ps += mean_sz_i[i] * mean_sz_i[j]
        end
        mean_prod_sz[r] = ps / L
    end
    
    obs.final_corr .= avg_szsz_r .- mean_prod_sz
end

#--------------------------------------------------------------------
# Observable: Sz-Sz Time Correlation
# C(t) = <Sz(t)Sz(0)> - <Sz(t)><Sz(0)> (spatially averaged)
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
    for idx in 1:length(obs.tmea)
        num_meas = obs.num_lag_measurements[idx]
        if num_meas > 0
            avg_prod = obs.prod_sum[idx] / num_meas
            avg_mean_t = obs.mean_t_sum[idx] / num_meas
            avg_mean_0 = obs.mean_0_sum[idx] / num_meas
            obs.final_corr[idx] = avg_prod - (avg_mean_t * avg_mean_0)
        end
    end
end


#--------------------------------------------------------------------
# Observable: O Operator Expectation Value <O>
#--------------------------------------------------------------------
mutable struct OExpectation <: AbstractObservable
    sum::Float64
    final_value::Float64
    OExpectation() = new(0.0, 0.0)
end

get_filename(obs::OExpectation) = "o_expectation.dat"
get_header(obs::OExpectation) = "Observable\tAvg_Value\tErr_Value"
get_data(obs::OExpectation) = [obs.final_value]
# For single value observables, xaxis can be a descriptive array
get_xaxis(obs::OExpectation, L::Int, tmea::Vector{Int}) = ["<O>"]

function measure!(obs::OExpectation, sys, workspace)
    obs.sum += sum(workspace.O_vals) / sys.L
end

function finalize!(obs::OExpectation, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end


#--------------------------------------------------------------------
# Observable: Equal-Time O-O Spatial Correlation C_oo(r)
#--------------------------------------------------------------------
mutable struct OOSpatialCorr <: AbstractObservable
    corr_sum::Vector{Float64}
    mean_sum::Vector{Float64}
    final_corr::Vector{Float64}

    function OOSpatialCorr(L::Int)
        max_dist = L ÷ 2
        new(zeros(Float64, max_dist), zeros(Float64, L), zeros(Float64, max_dist))
    end
end

get_filename(obs::OOSpatialCorr) = "oo_r_corr.dat"
get_header(obs::OOSpatialCorr) = "r\tOO_r_Avg\tOO_r_Err"
get_data(obs::OOSpatialCorr) = obs.final_corr
get_xaxis(obs::OOSpatialCorr, L::Int, tmea::Vector{Int}) = 1:(L÷2)

function measure!(obs::OOSpatialCorr, sys, workspace)
    L = sys.L
    O_vals = workspace.O_vals
    max_dist = L ÷ 2
    
    obs.mean_sum .+= O_vals
    
    for r in 1:max_dist
        prod_sum = 0.0
        for i in 1:L
            j = (i + r - 1) % L + 1
            prod_sum += O_vals[i] * O_vals[j]
        end
        obs.corr_sum[r] += prod_sum / L
    end
end

function finalize!(obs::OOSpatialCorr, total_measurements::Int)
    if total_measurements == 0 return end
    
    L = length(obs.mean_sum)
    max_dist = length(obs.corr_sum)
    
    avg_oo_r = obs.corr_sum ./ total_measurements
    mean_o_i = obs.mean_sum ./ total_measurements

    mean_prod_o = zeros(Float64, max_dist)
    for r in 1:max_dist
        ps = 0.0
        for i in 1:L
            j = (i + r - 1) % L + 1
            ps += mean_o_i[i] * mean_o_i[j]
        end
        mean_prod_o[r] = ps / L
    end
    
    obs.final_corr .= avg_oo_r .- mean_prod_o
end


#--------------------------------------------------------------------
# Observable: O-O Time Correlation
# C(t) = <O(t)O(0)> - <O(t)><O(0)> (spatially averaged)
# This observable manages its own history buffer for O values.
#--------------------------------------------------------------------
mutable struct OOTCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    O_history::Matrix{Int} # Local history buffer
    
    prod_sum::Vector{Float64}
    mean_t_sum::Vector{Float64}
    mean_0_sum::Vector{Float64}
    num_lag_measurements::Vector{Int}
    final_corr::Vector{Float64}

    function OOTCorr(tmea::Vector{Int})
        num_lags = length(tmea)
        buffer_size = isempty(tmea) ? 1 : maximum(tmea) + 1
        # History buffer is initially empty, will be allocated in initialize!
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
    for idx in 1:length(obs.tmea)
        num_meas = obs.num_lag_measurements[idx]
        if num_meas > 0
            avg_prod = obs.prod_sum[idx] / num_meas
            avg_mean_t = obs.mean_t_sum[idx] / num_meas
            avg_mean_0 = obs.mean_0_sum[idx] / num_meas
            obs.final_corr[idx] = avg_prod - (avg_mean_t * avg_mean_0)
        end
    end
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
            
            obs.h_sq_diff_sum[idx] += (current_H_mid - past_H_mid)^2
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
# Observable: Spin Magnitude Autocorrelation
# P(|S^z(t)|=|S^z(0)|, |S^z(0)|!=0)
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

####### ########
get_filename(obs::SpinMagnitudeAutocorr) = "spin_mag_autocorr.dat"
get_header(obs::SpinMagnitudeAutocorr) = "t\tP_Avg\tP_Err"
get_data(obs::SpinMagnitudeAutocorr) = obs.final_corr
get_xaxis(obs::SpinMagnitudeAutocorr, L::Int, tmea::Vector{Int}) = tmea

function measure!(obs::SpinMagnitudeAutocorr, sys, workspace)
    # This observable needs the global Z_history, which is managed by the system
    if !sys.use_history
        error("SpinMagnitudeAutocorr requires the system's history buffer to be enabled.")
    end

    current_buffer_idx = (workspace.t - 1) % sys.buffer_size + 1
    current_Z = view(sys.Z_history, :, current_buffer_idx)

    for (idx, lag) in enumerate(obs.tmea)
        if workspace.t - lag > sys.Tthermal
            past_buffer_idx = (workspace.t - lag - 1) % sys.buffer_size + 1
            past_Z = view(sys.Z_history, :, past_buffer_idx)
            
            # Iterate over all sites to accumulate statistics
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
            obs.final_corr[idx] = 0.0 # Or NaN, depending on desired output for no data
        end
    end
end

# observables.jl (replace the P1Autocorr block with this)


# observables.jl (add this new observable)

#--------------------------------------------------------------------
# Observable: General Pm Projector Expectation Value <Pm>
# <Pm> = < 1/L * sum_i Pm(i) >
# where Pm(i) = 1 if |Sz(i)| == m, and 0 otherwise.
#--------------------------------------------------------------------
mutable struct PmExpectation <: AbstractObservable
    m::Int
    sum::Float64
    final_value::Float64

    function PmExpectation(m::Int)
        if m < 0
            error("Spin magnitude 'm' for PmExpectation must be a positive integer.")
        end
        new(m, 0.0, 0.0)
    end
end

get_filename(obs::PmExpectation) = "pm$(obs.m)_expectation.dat"
get_header(obs::PmExpectation) = "m\tPm_Avg\tPm_Err"

# This observable produces a single value. We return it as a 1-element vector
# to conform to the generic gather_process_save function.
get_data(obs::PmExpectation) = [obs.final_value]
get_xaxis(obs::PmExpectation, L::Int, tmea::Vector{Int}) = [obs.m]

function measure!(obs::PmExpectation, sys, workspace)
    # Create a boolean vector where the condition |Sz|==m is met
    p_m = abs.(sys.BracketConfig) .== obs.m
    
    # Add the spatial average for the current step to the total sum
    obs.sum += sum(p_m) / sys.L
end

function finalize!(obs::PmExpectation, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: General Pm Projector Autocorrelation
# C(t) = <Pm(t)Pm(0)> - <Pm(t)><Pm(0)>
# where Pm(i) = 1 if |Sz(i)| == m, and 0 otherwise.
# 'm' is a parameter specified at construction.
#--------------------------------------------------------------------
mutable struct PmAutocorr <: AbstractObservable
    m::Int # The spin magnitude to project onto
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

# Filename and header are now dynamically generated based on 'm'
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
            
            # The check is now generalized to obs.m instead of a hardcoded 1
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



# observables.jl (add this new observable)

#--------------------------------------------------------------------
# Observable: General Pm Projector Expectation Value <Pm>
# <Pm> = < 1/L * sum_i Pm(i) >
# where Pm(i) = 1 if Sz(i) == m, and 0 otherwise.
#--------------------------------------------------------------------
mutable struct NmExpectation <: AbstractObservable
    m::Int
    sum::Float64
    final_value::Float64

    function NmExpectation(m::Int)
        new(m, 0.0, 0.0)
    end
end

get_filename(obs::NmExpectation) = "nm$(obs.m)_expectation.dat"
get_header(obs::NmExpectation) = "m\tNm_Avg\tNm_Err"

# This observable produces a single value. We return it as a 1-element vector
# to conform to the generic gather_process_save function.
get_data(obs::NmExpectation) = [obs.final_value]
get_xaxis(obs::NmExpectation, L::Int, tmea::Vector{Int}) = [obs.m]

function measure!(obs::NmExpectation, sys, workspace)
    # Create a boolean vector where the condition |Sz|==m is met
    n_m = (sys.BracketConfig) .== obs.m
    
    # Add the spatial average for the current step to the total sum
    obs.sum += sum(n_m) / sys.L
end

function finalize!(obs::NmExpectation, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end

#--------------------------------------------------------------------
# Observable: General Pm Projector Autocorrelation
# C(t) = <Pm(t)Pm(0)> - <Pm(t)><Pm(0)>
# where Pm(i) = 1 if Sz(i) == m, and 0 otherwise.
# 'm' is a parameter specified at construction.
#--------------------------------------------------------------------
mutable struct NmAutocorr <: AbstractObservable
    m1::Int # The spin magnitude to project onto
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

# Filename and header are now dynamically generated based on 'm'
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
            
            # The check is now generalized to obs.m instead of a hardcoded 1
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


# observables.jl (add these new observable blocks)

#--------------------------------------------------------------------
# Observable: C Operator Expectation Value <C>
#--------------------------------------------------------------------
mutable struct CExpectation <: AbstractObservable
    sum::Float64
    final_value::Float64
    CExpectation() = new(0.0, 0.0)
end

get_filename(obs::CExpectation) = "c_expectation.dat"
get_header(obs::CExpectation) = "Observable\tAvg_Value\tErr_Value"
get_data(obs::CExpectation) = [obs.final_value]
get_xaxis(obs::CExpectation, L::Int, tmea::Vector{Int}) = ["<C>"]

function measure!(obs::CExpectation, sys, workspace)
    # workspace.C_vals is pre-calculated in the main loop
    obs.sum += sum(workspace.C_vals) / sys.L
end

function finalize!(obs::CExpectation, total_measurements::Int)
    if total_measurements > 0
        obs.final_value = obs.sum / total_measurements
    end
end


#--------------------------------------------------------------------
# Observable: C-C Time Correlation
# C(t) = <C(t)C(0)> - <C(t)><C(0)> (spatially averaged)
# This observable manages its own history buffer for C values.
#--------------------------------------------------------------------
mutable struct CCTCorr <: AbstractObservable
    tmea::Vector{Int}
    buffer_size::Int
    C_history::Matrix{Int} # Local history buffer
    
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
    # Allocate local history buffer
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
    for idx in 1:length(obs.tmea)
        num_meas = obs.num_lag_measurements[idx]
        if num_meas > 0
            avg_prod = obs.prod_sum[idx] / num_meas
            avg_mean_t = obs.mean_t_sum[idx] / num_meas
            avg_mean_0 = obs.mean_0_sum[idx] / num_meas
            obs.final_corr[idx] = avg_prod - (avg_mean_t * avg_mean_0)
        end
    end
end