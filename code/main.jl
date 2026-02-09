
using Random, Dates, Serialization
using LinearAlgebra
using MPI
using Statistics


include("Observables.jl")
using .Observables
include("sub.jl")
include("utils.jl")
include("config.jl")
const BASE_DIR = dirname(@__FILE__) # main.jl 所在的目录
# Auto-load all observables from the directory.
load_observables_from_directory!(joinpath(BASE_DIR, "Observables"))

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
    period_idx = parse(Int,ARGS[6])
    task = ARGS[7]
    # modes = Set(split(lowercase(mode_str), ','))
    # run_all = "all" in modes

    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    MPIsize = MPI.Comm_size(comm)


    # --- Build the list of observables to run based on modes ---
    observables_to_run = AbstractObservable[]
    
    # # # # static observables
    
    # push!(observables_to_run, create_observable("o_expect", L, Int64[]))
    # push!(observables_to_run, create_observable("q_expect", L, Int64[]))
    # push!(observables_to_run, create_observable("nm_expect", L, Int64[]; m=0))
    # push!(observables_to_run, create_observable("oo_tcorr", L, tmea))
    # push!(observables_to_run, create_observable("qq_tcorr", L, tmea))
    
    # --- Check for time-dependent observables to determine if history buffer is needed ---
    z=2.7; total_points=200
    periods = get_periods(L, PERIOD_CONFIG)
    

    

    interval = periods[period_idx][2]
    # needs_time_obs = run_all || any(startswith(m, "pm") for m in modes) in modes # Add other time obs here

    tmea0 = generate_log_spaced_times(L, z, total_points, periods; 
                                    max_scaled_time=0.5,
                                    period_index=period_idx)                        
    tmea = Int64.(tmea0/interval)  

    if task == "basic"
        push!(observables_to_run, create_observable("sz_expect", L, Int64[]))
        push!(observables_to_run, create_observable("h_expect", L, Int64[]))
        push!(observables_to_run, create_observable("szsz_tcorr", L, tmea))
        push!(observables_to_run, create_observable("hh_tcorr", L, tmea))
        push!(observables_to_run, create_observable("hd_tcorr", L, tmea))
    elseif task == "abcd"
        push!(observables_to_run, create_observable("a_expect", L, Int64[]))
        push!(observables_to_run, create_observable("b_expect", L, Int64[]))
        push!(observables_to_run, create_observable("aa_tcorr", L, tmea))
        push!(observables_to_run, create_observable("bb_tcorr", L, tmea))
        if S >= 2
            push!(observables_to_run, create_observable("c_expect", L, Int64[]))
            push!(observables_to_run, create_observable("d_expect", L, Int64[]))
            push!(observables_to_run, create_observable("cc_tcorr", L, tmea))
            push!(observables_to_run, create_observable("dd_tcorr", L, tmea))
        end
    elseif task == "ab"
        push!(observables_to_run, create_observable("a_expect", L, Int64[]))
        push!(observables_to_run, create_observable("b_expect", L, Int64[]))
        push!(observables_to_run, create_observable("aa_tcorr", L, tmea))
        push!(observables_to_run, create_observable("bb_tcorr", L, tmea))
    elseif task == "a"
        push!(observables_to_run, create_observable("a_expect", L, Int64[]))
        push!(observables_to_run, create_observable("aa_tcorr", L, tmea))
        # push!(observables_to_run, create_observable("b_expect", L, Int64[]))
    elseif task == "cd"
        push!(observables_to_run, create_observable("c_expect", L, Int64[]))
        push!(observables_to_run, create_observable("d_expect", L, Int64[]))
        push!(observables_to_run, create_observable("cc_tcorr", L, tmea))
        push!(observables_to_run, create_observable("dd_tcorr", L, tmea))
    elseif task == "oq"
        # push!(observables_to_run, create_observable("omn_expect", L, Int64[]; m=1, n = 1))
        # push!(observables_to_run, create_observable("omn_tcorr", L, tmea; m =1, n = 1))
        # push!(observables_to_run, create_observable("qmn_expect", L, Int64[]; m=1, n = 1))
        # push!(observables_to_run, create_observable("qmn_tcorr", L, tmea; m =1, n = 1))
        push!(observables_to_run, create_observable("omnqmn_tcorr", L, tmea; m =1, n = 1))
        
    elseif task == "o"
        push!(observables_to_run, create_observable("omn_expect", L, Int64[]; m=1, n = 1))
        push!(observables_to_run, create_observable("omn_tcorr", L, tmea; m =1, n = 1))
    elseif task == "q"
        push!(observables_to_run, create_observable("qmn_expect", L, Int64[]; m=1, n = 1))
        push!(observables_to_run, create_observable("qmn_tcorr", L, tmea; m =1, n = 1))
    elseif task == "all"
        push!(observables_to_run, create_observable("sz_expect", L, Int64[]))
        push!(observables_to_run, create_observable("h_expect", L, Int64[]))
        push!(observables_to_run, create_observable("szsz_tcorr", L, tmea))
        push!(observables_to_run, create_observable("hh_tcorr", L, tmea))
        push!(observables_to_run, create_observable("hd_tcorr", L, tmea))

        push!(observables_to_run, create_observable("a_expect", L, Int64[]))
        push!(observables_to_run, create_observable("b_expect", L, Int64[]))
        push!(observables_to_run, create_observable("aa_tcorr", L, tmea))
        push!(observables_to_run, create_observable("bb_tcorr", L, tmea))
        if S >= 2
            push!(observables_to_run, create_observable("c_expect", L, Int64[]))
            push!(observables_to_run, create_observable("d_expect", L, Int64[]))
            push!(observables_to_run, create_observable("cc_tcorr", L, tmea))
            push!(observables_to_run, create_observable("dd_tcorr", L, tmea))
        end

        push!(observables_to_run, create_observable("omn_expect", L, Int64[]; m=1, n = 1))
        push!(observables_to_run, create_observable("omn_tcorr", L, tmea; m =1, n = 1))

        push!(observables_to_run, create_observable("qmn_expect", L, Int64[]; m=1, n = 1))
        push!(observables_to_run, create_observable("qmn_tcorr", L, tmea; m =1, n = 1))
        
        push!(observables_to_run, create_observable("omnqmn_tcorr", L, tmea; m =1, n = 1))
            
    end

    needs_time_obs = !isempty(tmea)
    if rank == 0
        println("--- Active Observables ---")
        for obs in observables_to_run
            println("- $(typeof(obs)) -> $(get_filename(obs))")
        end
        println("--------------------------")
    end
    
    output_dir = rank == 0 ? create_output_dir(L, S, T, Tthermal, run_id; period_index = period_idx) : ""
    output_dir = MPI.bcast(output_dir, comm, root=0)
    
    seed = abs(rand(Int))
    println("[Rank $rank] Using random seed = $seed")
    Random.seed!(seed)

    # The system is created knowing if any time-displaced observables are active.
    sys = BracketSystem(L, Int64(T/interval), Int64(Tthermal/interval), tmea, seed, rank, S, needs_time_obs)
    InitialConfigSpinS!(sys)

    start_time = time()
    
    RunAndMeasure!(sys, interval, observables_to_run, tmea)
    println("[Rank $rank] Simulation finished in: $(round(time() - start_time, digits=2)) seconds")
    
    # --- Data-Driven Post-Processing ---
    for obs in observables_to_run
        gather_process_save(obs, comm, output_dir, L, tmea0)
    end

    if rank == 0
        open(joinpath(output_dir, "parameters.txt"), "w") do io
            println(io, "L = $L\nS = $S\nT = $T\nTthermal = $Tthermal\nrun_id = $run_id\nperiod_idx = $period_idx\ninterval = $interval\nnum_processes = $MPIsize")
        end
        println("[Rank 0] All tasks complete. Results saved to $output_dir")
    end
    
    MPI.Finalize()
end

main()