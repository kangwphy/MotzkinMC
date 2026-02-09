function create_output_dir(L::Int, S::Int, T::Int, Tthermal::Int, run_id::Int; period_index::Union{Int,Nothing} = nothing)
    if period_index !== nothing
        dir_name = "data/L$(L)_S$(S)_T$(T)_Tthermal$(Tthermal)/run$(run_id)/period$(period_index)"
    else
        dir_name = "data/L$(L)_S$(S)_T$(T)_Tthermal$(Tthermal)/run$(run_id)"
    end
    mkpath(dir_name)
    println("Creating output directory: $dir_name")
    return dir_name
end

# function generate_log_spaced_times(L::Int, z::Float64, num_points::Int; max_scaled_time::Float64=0.5, t_min::Int=1)
#     t_max = max_scaled_time * (L ^ z)
#     return isempty(t_max) || t_max < t_min ? [t_min] : unique(round.(Int, 10 .^ range(log10(t_min), stop=log10(t_max), length=num_points)))
# end

"""
    generate_log_spaced_times(...)

Generates logarithmically spaced measurement times, with options to select a specific
sub-period for execution.

# Arguments
- `periods`: A vector of tuples `(t_threshold, interval)`.
- `period_index` (optional): An integer `n` specifying which period to generate times for.
  If `nothing` (the default), times are generated for all periods.
"""
function generate_log_spaced_times(
    L::Int, 
    z::Float64, 
    num_points::Int,
    periods::Vector; 
    max_scaled_time::Float64=0.5, 
    t_min::Int=1,
    period_index::Union{Int, Nothing}=nothing
)
    # Determine the absolute maximum time for the whole simulation
    t_max_global = max_scaled_time * (L^z)
    if t_max_global < t_min
        return t_min > 0 ? [t_min] : Int[]
    end

    # 1. Determine the time boundaries [t_start, t_end) for the selected period
    local_t_start = t_min
    local_t_end = t_max_global

    if period_index !== nothing
        if !(1 <= period_index <= length(periods))
            error("period_index must be between 1 and $(length(periods)), or be `nothing`.")
        end
        
        # The end of the current period is its threshold
        local_t_end = periods[period_index][1]

        # The start of the current period is the threshold of the previous one
        if period_index > 1
            local_t_start = periods[period_index - 1][1]
        end
    end

    # 2. Generate the "ideal" log-spaced points across the ENTIRE global range
    #    This is crucial to maintain the correct logarithmic density of points.
    log_steps = range(log10(t_min), stop=log10(t_max_global), length=num_points)
    tmea_float = 10 .^ log_steps

    snapped_times = Int[]
    for t in tmea_float
        # 3. Filter out points that are not in our selected period
        if t < local_t_start || t >= local_t_end
            continue
        end

        # 4. Find the correct interval for this point's period
        current_interval = 1
        for (threshold, intrvl) in periods
            if t < threshold
                current_interval = intrvl
                break
            end
        end
        
        # 5. Snap the ideal point to the nearest multiple of the interval
        snapped_t = round(Int, t / current_interval) * current_interval
        
        # Ensure the snapped time is valid and within the period's bounds
        if snapped_t >= local_t_start
             push!(snapped_times, max(t_min, snapped_t))
        end
    end

    # 6. Return the unique, sorted list of valid measurement times
    return sort(unique(snapped_times))
end

# This function is now generic and works for any object that conforms to the AbstractObservable interface.
function gather_process_save(obs::AbstractObservable, comm::MPI.Comm, output_dir::String, L::Int, tmea::Vector{Int})
    rank = MPI.Comm_rank(comm)
    MPIsize = MPI.Comm_size(comm)
    
    data_to_gather = get_data(obs)
    if isempty(data_to_gather) return end
    
    gathered_data = rank == 0 ? zeros(Float64, length(data_to_gather), MPIsize) : nothing
    MPI.Gather!(data_to_gather, gathered_data, comm, root=0)

    if rank == 0
        xaxis = get_xaxis(obs, L, tmea)
        avg = [mean(view(gathered_data, i, :)) for i in 1:length(xaxis)]
        err = [std(view(gathered_data, i, :)) / sqrt(MPIsize) for i in 1:length(xaxis)]
        
        output_path = joinpath(output_dir, get_filename(obs))
        open(output_path, "w") do io
            println(io, get_header(obs))
            for (idx, x_val) in enumerate(xaxis)
                println(io, "$x_val\t$(avg[idx])\t$(err[idx])")
            end
        end
        println("[Rank 0] Saved: $(basename(output_path))")
    end
end