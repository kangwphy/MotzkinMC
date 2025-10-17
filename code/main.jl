
using Random, Dates, Serialization
using LinearAlgebra
using MPI
using Statistics


include("Observables.jl")
using .Observables
include("sub.jl")
const BASE_DIR = dirname(@__FILE__) # main.jl 所在的目录
# Auto-load all observables from the directory.
load_observables_from_directory!(joinpath(BASE_DIR, "Observables"))
function create_output_dir(L::Int, S::Int, T::Int, Tthermal::Int, run_id::Int)
    dir_name = "data/L$(L)_S$(S)_T$(T)_Tthermal$(Tthermal)/run$(run_id)"
    mkpath(dir_name)
    println("Creating output directory: $dir_name")
    return dir_name
end

function generate_log_spaced_times(L::Int, z::Float64, num_points::Int; max_scaled_time::Float64=0.5, t_min::Int=1)
    t_max = max_scaled_time * (L ^ z)
    return isempty(t_max) || t_max < t_min ? [t_min] : unique(round.(Int, 10 .^ range(log10(t_min), stop=log10(t_max), length=num_points)))
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

function main()
    if length(ARGS) < 6
        println("Usage: mpiexec -n <procs> julia main.jl L T Tthermal run_id S mode")
        println("Example: mpiexec -n 4 julia main.jl 100 10000 1000 1 2 all")
        println("Available modes: 'all', 'sz_profile', 'szsz_r', 'height_diff', etc. Combine with commas.")
        return
    end

    L=parse(Int,ARGS[1])
    T=parse(Int,ARGS[2])
    Tthermal=parse(Int,ARGS[3])
    run_id=parse(Int,ARGS[4])
    S=parse(Int,ARGS[5])
    mode_str=ARGS[6]
    
    modes = Set(split(lowercase(mode_str), ','))
    run_all = "all" in modes

    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    MPIsize = MPI.Comm_size(comm)


    # --- Build the list of observables to run based on modes ---
    observables_to_run = AbstractObservable[]
    
    # # # static observables
    push!(observables_to_run, create_observable("sz_expect", L, Int64[]))
    push!(observables_to_run, create_observable("h_expect", L, Int64[]))
    push!(observables_to_run, create_observable("o_expect", L, Int64[]))
    push!(observables_to_run, create_observable("q_expect", L, Int64[]))
    push!(observables_to_run, create_observable("nm_expect", L, Int64[]; m=0))
    
    
    # --- Check for time-dependent observables to determine if history buffer is needed ---
    tmea = Int[]
    z=2.7; total_points=200
    needs_time_obs = run_all || any(startswith(m, "pm") for m in modes) in modes # Add other time obs here
    tmea = generate_log_spaced_times(L, z, total_points; max_scaled_time=min(10000000/(L^z), 0.5))

    push!(observables_to_run, create_observable("szsz_tcorr", L, tmea))
    push!(observables_to_run, create_observable("oo_tcorr", L, tmea))
    push!(observables_to_run, create_observable("qq_tcorr", L, tmea))
    push!(observables_to_run, create_observable("h_tcorr", L, tmea))
    push!(observables_to_run, create_observable("h_diffusion", L, tmea))

    if S >= 2
        push!(observables_to_run, create_observable("a_expect", L, Int64[]))
        push!(observables_to_run, create_observable("b_expect", L, Int64[]))
        push!(observables_to_run, create_observable("c_expect", L, Int64[]))
        push!(observables_to_run, create_observable("d_expect", L, Int64[]))
        push!(observables_to_run, create_observable("aa_tcorr", L, tmea))
        push!(observables_to_run, create_observable("bb_tcorr", L, tmea))
        push!(observables_to_run, create_observable("cc_tcorr", L, tmea))
        push!(observables_to_run, create_observable("dd_tcorr", L, tmea))
    end


    if rank == 0
        println("--- Active Observables ---")
        for obs in observables_to_run
            println("- $(typeof(obs)) -> $(get_filename(obs))")
        end
        println("--------------------------")
    end
    
    output_dir = rank == 0 ? create_output_dir(L, S, T, Tthermal, run_id) : ""
    output_dir = MPI.bcast(output_dir, comm, root=0)
    
    seed = abs(rand(Int))
    println("[Rank $rank] Using random seed = $seed")
    Random.seed!(seed)

    # The system is created knowing if any time-displaced observables are active.
    sys = BracketSystem(L, T, Tthermal, tmea, seed, rank, S, needs_time_obs)
    InitialConfigSpinS!(sys)
    
    start_time = time()
    interval = 1
    RunAndMeasure!(sys, interval, observables_to_run, tmea)
    println("[Rank $rank] Simulation finished in: $(round(time() - start_time, digits=2)) seconds")
    
    # --- Data-Driven Post-Processing ---
    for obs in observables_to_run
        gather_process_save(obs, comm, output_dir, L, tmea)
    end

    if rank == 0
        open(joinpath(output_dir, "parameters.txt"), "w") do io
            println(io, "L = $L\nS = $S\nT = $T\nTthermal = $Tthermal\nrun_id = $run_id\nmode = $mode_str\ninterval = $interval\nnum_processes = $MPIsize")
        end
        println("[Rank 0] All tasks complete. Results saved to $output_dir")
    end
    
    MPI.Finalize()
end

main()