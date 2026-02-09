using DelimitedFiles
using Glob
using Printf
using Statistics

# ==============================================================================
# --- Function to Merge Period Data within a Single Run (MODIFIED) ---
# ==============================================================================

"""
    merge_periods_in_run(run_dir::String)

Merges data from `period*` subdirectories into the parent `run_dir`.
- For `_tcorr.dat` files: Concatenates and sorts by time.
- For `_expect.dat` files: Copies the file from the first period only.
- For `parameters.txt`: Copies the file from the first period.
"""
function merge_periods_in_run(run_dir::String)
    println("  Scanning for periods in: $(basename(run_dir))")
    period_dirs = glob(joinpath(run_dir, "period*"))

    # Sort directories to reliably find the first one (e.g., period1)
    try
        sort!(period_dirs, by = x -> parse(Int, match(r"period(\d+)", basename(x)).captures[1]))
    catch e
        @warn "Could not sort period directories numerically in '$run_dir'. Using default order."
    end

    if isempty(period_dirs)
        println("    - No 'period*' subdirectories found. Skipping merge.")
        return
    end

    first_period_dir = period_dirs[1]

    # --- NEW: 1. Copy parameters.txt from the first period ---
    source_params = joinpath(first_period_dir, "parameters.txt")
    dest_params = joinpath(run_dir, "parameters.txt")
    if isfile(source_params)
        cp(source_params, dest_params, force=true)
        println("    - Copied 'parameters.txt' from $(basename(first_period_dir))")
    end

    # Group all other .dat files by name across all period directories
    files_to_merge = Dict{String, Vector{String}}()
    for p_dir in period_dirs
        if !isdir(p_dir) continue end
        for filepath in glob(joinpath(p_dir, "*.dat"))
            filename = basename(filepath)
            if !haskey(files_to_merge, filename)
                files_to_merge[filename] = String[]
            end
            push!(files_to_merge[filename], filepath)
        end
    end

    if isempty(files_to_merge)
        println("    - No .dat files found in period directories.")
        return
    end

    # Process each group of files
    for (filename, path_list) in files_to_merge
        if isempty(path_list) continue end

        # --- NEW: 2. Special handling for _expect.dat files ---
        if occursin("_expect.dat", filename)
            source_expect_file = joinpath(first_period_dir, filename)
            if isfile(source_expect_file)
                dest_expect_file = joinpath(run_dir, "$(filename)")
                cp(source_expect_file, dest_expect_file, force=true)
                println("    - Copied (not merged) '$filename' from $(basename(first_period_dir))")
            end
            continue # Skip to the next file
        end

        # --- Standard merging logic for correlation files ---
        all_data = nothing
        header = ""
        try
            for (i, filepath) in enumerate(path_list)
                if !isfile(filepath) continue end
                if i == 1; header = readline(filepath); end
                
                current_data = readdlm(filepath, skipstart=1, comments=true)
                if isempty(current_data) continue end
                
                all_data = (all_data === nothing) ? current_data : vcat(all_data, current_data)
            end

            if all_data !== nothing && !isempty(all_data)
                sorted_data = sortslices(all_data, dims=1, by=x->x[1])
                unique_sorted_data = sorted_data[unique(i -> sorted_data[i,1], 1:size(sorted_data,1)), :]

                output_path = joinpath(run_dir, "$(filename)")
                open(output_path, "w") do io
                    println(io, header)
                    writedlm(io, unique_sorted_data, '\t')
                end
                println("    -> Merged and sorted $(length(path_list)) period files into $(basename(output_path))")
            end
        catch e
            @warn "Could not merge files for '$filename' in '$run_dir'. Error: $e"
        end
    end
end


# ==============================================================================
# --- GLOBAL CONFIGURATION FOR FILE HEADERS ---
# ==============================================================================
const FILE_HEADERS = Dict{String, Any}(
    "hh_tcorr.dat"     => "t HH_t_Avg HH_t_Err",
    "oo_tcorr.dat"     => "t OO_t_Avg OO_t_Err",
    "qq_tcorr.dat"     => "t QQ_t_Avg QQ_t_Err",
    "szsz_tcorr.dat"   => "t SzSz_t_Avg SzSz_t_Err",
    "hd_tcorr.dat"     => "t HD_Avg HD_Err",
    "aa_tcorr.dat"     => "t AA_t_Avg AA_t_Err",
    "bb_tcorr.dat"     => "t BB_t_Avg BB_t_Err",
    "cc_tcorr.dat"     => "t CC_t_Avg CC_t_Err",
    "dd_tcorr.dat"     => "t DD_t_Avg DD_t_Err",
    "a_expect.dat"     => ["A_Avg"],
    "b_expect.dat"     => ["B_Avg"],
    "c_expect.dat"     => ["C_Avg"],
    "d_expect.dat"     => ["D_Avg"],
    "q_expect.dat"     => ["Q_Avg"],
    "o_expect.dat"     => ["O_Avg"],
    "h_expect.dat"     => ["H_Avg"],
    "sz_expect.dat"    => ["Sz_Avg"],
)

const FINAL_OUTPUT_MAPPING = Dict(
    :time_corr      => "final_grand_average_time.dat",
    :equal_corr     => "final_grand_average_equal.dat",
    :local_obs      => "final_grand_average_obs.dat",
)

# ==============================================================================
# --- Data Reading and Processing Functions ---
# ==============================================================================

function read_correlations(filepath::String)
    data = readdlm(filepath, skipstart=1, comments=true)
    if isempty(data) return Float64[], Matrix{Float64}(undef, 0,0), Matrix{Float64}(undef, 0,0) end
    t_values = data[:, 1]
    mean_values = data[:, 2:2:end]
    error_values = data[:, 3:2:end]
    return t_values, mean_values, error_values
end

# function read_profile(filepath::String)
#     data = readdlm(filepath, skipstart=1, comments=true)
#     if isempty(data) return Float64[], Float64[] end
#     num_cols = size(data, 2)
#     num_observables = Int((num_cols) / 2) # Simplified for single column of data
    
#     if num_observables == 0 return Float64[], Float64[] end

#     avg_values = zeros(num_observables); avg_errors = zeros(num_observables)
    
#     for i in 1:num_observables
#         val_col = 2 * i - 1
#         err_col = 2 * i
        
#         # For expect files copied from one run, we don't average over sites.
#         # We assume it's a single value. If it's a profile, this needs adjustment.
#         # For now, let's assume it's a single scalar value.
#         avg_values[i] = data[1, val_col]
#         avg_errors[i] = data[1, err_col]
#     end

#     return avg_values, avg_errors
# end

function read_profile(filepath::String)
    data = readdlm(filepath, skipstart=1, comments=true)
    if isempty(data) return Float64[], Float64[] end
    
    num_cols = size(data, 2)
    # CORRECTED: This formula correctly accounts for the first column being a label/site index.
    num_observables = Int((num_cols - 1) / 2)
    
    if num_observables == 0 return Float64[], Float64[] end

    avg_values = zeros(num_observables)
    avg_errors = zeros(num_observables)
    
    # Check if this is a scalar file (one data row) or a profile file (multiple rows)
    if size(data, 1) == 1
        # --- SCALAR LOGIC ---
        # For files like a_expect.dat, just read the single value.
        for i in 1:num_observables
            val_col = 1 + (2 * i - 1)
            err_col = 1 + (2 * i)
            avg_values[i] = data[1, val_col]
            avg_errors[i] = data[1, err_col]
        end
    else
        # --- PROFILE LOGIC ---
        # For files like h_expect.dat, average over all sites.
        num_sites = size(data, 1)
        for i in 1:num_observables
            val_col = 1 + (2 * i - 1)
            err_col = 1 + (2 * i)
            
            avg_values[i] = mean(data[:, val_col])
            # Correct error propagation for an averaged profile
            sum_sq_err = sum(data[:, err_col].^2)
            avg_errors[i] = sqrt(sum_sq_err) / num_sites
        end
    end

    return avg_values, avg_errors
end
function process_file_group(L::Int, S::Int, T::Int, Tthermal::Int, run_ids::Vector{Int}, file_list::Vector{String})
    dir = "data/L$(L)_S$(S)_T$(T)_Tthermal$(Tthermal)"
    results = Dict{Symbol, Dict{String, Any}}()
    
    for data_filename in file_list
        is_profile_scalar = occursin("_expect.dat", data_filename)
        read_func = is_profile_scalar ? read_profile : read_correlations
        
        tmea = nothing; total_data = nothing; total_squared_errors = nothing; total_ranks = 0
        
        for run_id in run_ids
            run_dir_pattern = joinpath(dir, "run$(run_id)*")
            matching_run_dirs = glob(run_dir_pattern)
            if isempty(matching_run_dirs) continue end
            run_dir = matching_run_dirs[1]

            params_file = joinpath(run_dir, "parameters.txt")
            if !isfile(params_file) continue end
            
            merged_path = joinpath(run_dir, "$(data_filename)")
            original_path = joinpath(run_dir, data_filename)
            data_file_to_read = isfile(merged_path) ? merged_path : original_path

            if !isfile(data_file_to_read) continue end

            try
                params = Dict{String, Any}()
                open(params_file, "r") do io
                    for line in readlines(io)
                        if occursin("=", line)
                            key, value = split(line, " = ")
                            params[strip(key)] = value
                        end
                    end
                end

                num_processes = haskey(params, "num_processes") ? parse(Int, params["num_processes"]) : 1
                total_ranks += num_processes

                current_tmea, current_data, current_errors = if is_profile_scalar
                    d, e = read_func(data_file_to_read)
                    Float64[], d, e
                else
                    read_func(data_file_to_read)
                end
                
                if isempty(current_data) continue end

                if total_data === nothing
                    tmea = current_tmea
                    total_data = current_data .* num_processes
                    total_squared_errors = (current_errors .* num_processes).^2
                else
                    if size(current_data) != size(total_data)
                        @warn "Data size mismatch for $data_filename in run $run_id. Skipping this run for this file."
                        total_ranks -= num_processes
                        continue
                    end
                    total_data .+= current_data .* num_processes
                    total_squared_errors .+= (current_errors .* num_processes).^2
                end
            catch e
                @warn "Failed to process $data_file_to_read. Error: $e"
            end
        end

        if total_data !== nothing && total_ranks > 0
            final_avg_data = total_data ./ total_ranks
            final_avg_errors = sqrt.(total_squared_errors) ./ total_ranks
            
            results[Symbol(data_filename)] = Dict("Ranks" => total_ranks, "tmea" => tmea, "avg" => final_avg_data, "err" => final_avg_errors)
            println("  ✅ Processed $data_filename (Total Ranks: $total_ranks)")
        end
    end
    return results
end

# ==============================================================================
# --- Final File Writing ---
# ==============================================================================

function write_local_output(dir::String, output_key::Symbol, file_results::Dict{Symbol, Dict{String, Any}})
    output_filename = get(FINAL_OUTPUT_MAPPING, output_key, "final_average_output.dat")
    
    if isempty(file_results)
        println("Skipping $output_filename: No data collected for this category.")
        return
    end

    open(joinpath(dir, output_filename), "w") do io
        
        if output_key == :local_obs
            all_obs_names = String[]
            for filename_symbol in keys(file_results)
                labels = get(FILE_HEADERS, String(filename_symbol), String[])
                for label in labels
                    obs_name = replace(label, "_Avg" => "")
                    if !(obs_name in all_obs_names)
                        push!(all_obs_names, obs_name)
                    end
                end
            end
            sort!(all_obs_names)

            println(io, "Observable Avg_Value Err_Value")
            
            for obs_name in all_obs_names
                found = false
                line_parts = Float64[] 

                target_filename_str = ""
                for (fname, flabels) in FILE_HEADERS
                    if isa(flabels, AbstractVector) && (obs_name * "_Avg" in flabels)
                        target_filename_str = fname
                        break
                    end
                end

                if target_filename_str != ""
                    target_filename_sym = Symbol(target_filename_str)
                    
                    if haskey(file_results, target_filename_sym)
                        file_data = file_results[target_filename_sym]
                        labels = get(FILE_HEADERS, target_filename_str, String[])
                        
                        local_obs_index = findfirst(isequal(obs_name * "_Avg"), labels)
                        
                        if local_obs_index !== nothing && local_obs_index <= length(file_data["avg"])
                            line_parts = [file_data["avg"][local_obs_index], file_data["err"][local_obs_index]]
                            found = true
                        end
                    end
                end
            
                if found
                    println(io, @sprintf("%s %.16f %.16f", obs_name, line_parts[1], line_parts[2]))
                else
                    println(io, @sprintf("%s %.16f %.16f", obs_name, NaN, NaN))
                end
            end

        else # VECTOR DATA LOGIC
            representative_file = Symbol("")
            files_to_merge = Symbol[]
            
            all_vector_candidates = [Symbol(k) for (k, v) in FILE_HEADERS if isa(v, String)]

            for file_sym in all_vector_candidates
                if haskey(file_results, file_sym)
                    if representative_file == Symbol(""); representative_file = file_sym; end
                    push!(files_to_merge, file_sym)
                end
            end
            
            if representative_file == Symbol(""); return; end
            unique!(files_to_merge)

            unified_header_parts = [output_key == :time_corr ? "t" : "r"]
            
            for file_sym in files_to_merge
                 header_str = get(FILE_HEADERS, String(file_sym), "")
                 parts = split(header_str)[2:end]
                 append!(unified_header_parts, parts)
            end
            
            println(io, join(unified_header_parts, " "))

            res_rep = file_results[representative_file]
            if res_rep["tmea"] === nothing || isempty(res_rep["tmea"]); return; end

            for i in 1:length(res_rep["tmea"])
                line_parts = [res_rep["tmea"][i]]
                for file_sym in files_to_merge
                    res = file_results[file_sym]
                    if i <= size(res["avg"], 1)
                        for j in 1:size(res["avg"], 2)
                            push!(line_parts, res["avg"][i, j])
                            push!(line_parts, res["err"][i, j])
                        end
                    else
                        num_cols_to_pad = length(split(get(FILE_HEADERS, String(file_sym), ""))) - 1
                        for _ in 1:num_cols_to_pad; push!(line_parts, NaN); end
                    end
                end
                println(io, join([@sprintf("%.16f", x) for x in line_parts], " "))
            end
        end
        println("✅ Successfully wrote final file to $output_filename")
    end
end


# ==============================================================================
# --- Execution Flow ---
# ==============================================================================

function collect_and_average_runs(L::Int, S::Int, T::Int, Tthermal::Int, run_ids::Vector{Int})
    time_corr_files = ["hh_tcorr.dat","hd_tcorr.dat","oo_tcorr.dat", "qq_tcorr.dat", "szsz_tcorr.dat", "aa_tcorr.dat", "bb_tcorr.dat", "cc_tcorr.dat", "dd_tcorr.dat"]
    equal_corr_files = ["hh_r_corr.dat", "oo_r_corr.dat", "szsz_r_corr.dat"]
    local_obs_files = ["h_expect.dat", "o_expect.dat", "q_expect.dat", "sz_expect.dat", "a_expect.dat", "b_expect.dat", "c_expect.dat", "d_expect.dat"]
    
    dir = "data/L$(L)_S$(S)_T$(T)_Tthermal$(Tthermal)"
    
    # Auto-discover new files from the first available run directory
    if !isempty(run_ids)
        first_run_dir = joinpath(dir, "run$(first(run_ids))")
        if isdir(first_run_dir)
            # Look for merged files first, as they are the source of truth
            for filepath in glob(joinpath(first_run_dir, "*.dat"))
                filename_original = replace(basename(filepath), "" => "")
                
                if occursin("tcorr.dat", filename_original) && !(filename_original in time_corr_files)
                    if !haskey(FILE_HEADERS, filename_original)
                        base_name = replace(filename_original, "_tcorr.dat" => "")
                        FILE_HEADERS[filename_original] = "t $(base_name)_t_Avg $(base_name)_t_Err"
                    end
                    if !(filename_original in time_corr_files); push!(time_corr_files, filename_original); end
                end
                if occursin("expect.dat", filename_original) && !(filename_original in local_obs_files)
                     if !haskey(FILE_HEADERS, filename_original)
                        base_name = replace(filename_original, "_expect.dat" => "")
                        FILE_HEADERS[filename_original] = ["$(base_name)_Avg"]
                    end
                    if !(filename_original in local_obs_files); push!(local_obs_files, filename_original); end
                end
            end
        end
    end

    time_results = process_file_group(L, S, T, Tthermal, run_ids, time_corr_files)
    equal_results = process_file_group(L, S, T, Tthermal, run_ids, equal_corr_files)
    local_obs_results = process_file_group(L, S, T, Tthermal, run_ids, local_obs_files)
    
    write_local_output(dir, :time_corr, time_results)
    write_local_output(dir, :equal_corr, equal_results)
    write_local_output(dir, :local_obs, local_obs_results)
end

function process_all_folders(base_dir::String)
    dir_pattern = r"L(\d+)_S(\d+)_T(\d+)_Tthermal(\d+)"
    
    for dir_path in glob(joinpath(base_dir, "*"))
        if !isdir(dir_path) continue end
        
        m = match(dir_pattern, basename(dir_path))
        if m === nothing continue end
        
        L, S, T, Tthermal = parse.(Int, m.captures)
        
        println("\n=========================================================")
        println("Processing parameter set: L=$L, S=$S, T=$T, Tthermal=$Tthermal")
        
        run_dirs_to_process = glob(joinpath(dir_path, "run*"))
        if isempty(run_dirs_to_process)
            println("  No 'run*' directories found to process.")
            continue
        end

        println("--- Merging period data for each run ---")
        for run_dir in run_dirs_to_process
            if isdir(run_dir)
                merge_periods_in_run(run_dir)
            end
        end
        println("--- Period merging complete ---")

        run_ids = [parse(Int, match(r"run(\d+)", basename(rd)).captures[1]) for rd in run_dirs_to_process if isdir(rd)]
        
        if !isempty(run_ids)
            println("\n--- Collecting and averaging across runs ---")
            collect_and_average_runs(L, S, T, Tthermal, sort(unique(run_ids)))
        end
        
        println("=========================================================")
    end
    
    println("\nFinished processing all data.")
end

# --- Script Execution ---
process_all_folders("data")