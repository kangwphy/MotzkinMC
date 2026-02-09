using DelimitedFiles
using Glob
using Printf 
using Statistics

# ==============================================================================
# --- GLOBAL CONFIGURATION FOR FILE HEADERS (Unchanged) ---
# ==============================================================================
const FILE_HEADERS = Dict(
    # Correlation Data (Vector Output)
    "hh_tcorr.dat"     => "t HH_t_Avg HH_t_Err",
    "oo_tcorr.dat"     => "t OO_t_Avg OO_t_Err",
    "qq_tcorr.dat"     => "t QQ_t_Avg QQ_t_Err",
    "szsz_tcorr.dat"   => "t SzSz_t_Avg SzSz_t_Err",
    "hd_tcorr.dat"     => "t HD_Avg HD_Err",
    "aa_tcorr.dat"     => "t AA_t_Avg AA_t_Err",
    "bb_tcorr.dat"     => "t BB_t_Avg BB_t_Err",
    "cc_tcorr.dat"     => "t CC_t_Avg CC_t_Err",
    "dd_tcorr.dat"     => "t DD_t_Avg DD_t_Err",
    "generic_tcorr.dat" => "t Corr_Avg Corr_Err", # 假设所有未明确列出的文件都只有一列数据

    # Profile/Scalar Data (Scalar Output)
    "a_expect.dat"     => ["A_Avg"],
    "b_expect.dat"     => ["B_Avg"],
    "c_expect.dat"     => ["C_Avg"],
    "d_expect.dat"     => ["D_Avg"],
    "q_expect.dat"     => ["Q_Avg"],
    "o_expect.dat"     => ["O_Avg"],
    "h_expect.dat"     => ["H_Avg"],
    "sz_expect.dat"    => ["Sz_Avg"],
    "generic_expect.dat" => ["Avg"], # 假设所有未明确列出的文件都只有一列数据
)

# --- Define the desired output filenames in the parameter folder (Unchanged) ---
const FINAL_OUTPUT_MAPPING = Dict(
    :time_corr      => "final_grand_average_time.dat",
    :equal_corr     => "final_grand_average_equal.dat",
    :local_obs      => "final_grand_average_obs.dat",
)

# ==============================================================================
# --- Data Reading and Processing Functions (Unchanged) ---
# ==============================================================================

function read_correlations(filepath::String)
    data = readdlm(filepath, skipstart=1, comments=true)
    t_values = data[:, 1]
    mean_values = data[:, 2:2:end]
    error_values = data[:, 3:2:end]
    return t_values, mean_values, error_values
end

function read_profile(filepath::String)
    data = readdlm(filepath, skipstart=1, comments=true)
    num_cols = size(data, 2)
    num_observables = Int((num_cols - 1) / 2)
    
    if num_observables == 0 return Float64[], Float64[] end

    avg_values = zeros(num_observables); avg_errors = zeros(num_observables)
    num_sites = size(data, 1)

    for i in 1:num_observables
        val_col = 2 * i      
        err_col = 2 * i + 1  
        
        avg_values[i] = mean(data[:, val_col])
        sum_sq_err = sum(data[:, err_col].^2)
        avg_errors[i] = sqrt(sum_sq_err) / num_sites
    end

    return avg_values, avg_errors
end

function process_file_group(L::Int, S::Int, T::Int, Tthermal::Int, run_ids::Vector{Int}, file_list::Vector{String}, result_key::Symbol)
    dir = "data/L$(L)_S$(S)_T$(T)_Tthermal$(Tthermal)"
    results = Dict{Symbol, Dict{String, Any}}()
    
    for data_filename in file_list
        is_profile_scalar = occursin("_expect.dat", data_filename)
        read_func = is_profile_scalar ? read_profile : read_correlations
        
        tmea = nothing; total_data = nothing; total_squared_errors = nothing; total_ranks = 0
        processed_runs = 0
        for run_id in run_ids
            try
                dir_pattern = joinpath(dir, "run$(run_id)*"); matching_dirs = glob(dir_pattern)
                if isempty(matching_dirs) continue end
                output_dir = matching_dirs[1]

                params_file = joinpath(output_dir, "parameters.txt"); if !isfile(params_file) continue end
                
                params = Dict{String, Any}()
                open(params_file, "r") do io
                    for line in readlines(io)
                        if occursin("=", line)
                            key, value = split(line, " = ")
                            params[strip(key)] = value
                        end
                    end
                end
   
                num_processes = parse(Int, params["num_processes"])
                total_ranks += num_processes

                data_file = joinpath(output_dir, data_filename)
                
                if isfile(data_file)
                    if is_profile_scalar
                        current_data, current_errors = read_func(data_file)
                        current_tmea = Float64[]
                    else
                        current_tmea, current_data, current_errors = read_func(data_file)
                    end
                    
                    if isempty(current_data) continue end

                    if total_data === nothing
                        tmea = current_tmea
                        total_data = current_data .* num_processes
                        total_squared_errors = (current_errors .* num_processes).^2
                    else
                        if size(current_data) != size(total_data) continue end
                        total_data .+= current_data .* num_processes
                        total_squared_errors .+= (current_errors .* num_processes).^2
                    end
                    processed_runs += 1
                end
            catch e
                continue
            end
        end

        if total_data === nothing || processed_runs == 0
            continue
        end

        final_avg_data = total_data ./ total_ranks
        final_avg_errors = sqrt.(total_squared_errors) ./ total_ranks
        
        results[Symbol(data_filename)] = Dict(
            "Ranks" => total_ranks,
            "tmea" => tmea, 
            "avg" => final_avg_data, 
            "err" => final_avg_errors,
        )
        
        println("  ✅ Processed $data_filename (Ranks: $total_ranks)")
    end
    return results
end

# ==============================================================================
# --- Final File Writing (MODIFIED TO FIX ERROR) ---
# ==============================================================================

function write_local_output(dir::String, output_key::Symbol, file_results::Dict{Symbol, Dict{String, Any}})
    output_filename = get(FINAL_OUTPUT_MAPPING, output_key, "final_average_output.dat")
    
    if isempty(file_results)
        println("Skipping $output_filename: No data collected for this category.")
        return
    end

    open(joinpath(dir, output_filename), "w") do io
        
        if output_key == :local_obs
            # --- SCALAR DATA LOGIC (Robust Implementation) ---
            
            all_obs_names = String[]
            for filename_symbol in keys(file_results)
                filename_str = String(filename_symbol)
                labels = get(FILE_HEADERS, filename_str, String[])
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
                    # *** FIX IS HERE ***
                    # Check if flabels is a collection (Vector) before using 'in'.
                    # This prevents the error when flabels is a String.
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

        else # --- VECTOR DATA LOGIC (Unchanged) ---
            
            representative_file = Symbol("")
            files_to_merge = Symbol[]
            
            all_vector_candidates = [Symbol(k) for (k, v) in FILE_HEADERS if isa(v, String)]

            for file_sym in all_vector_candidates
                if haskey(file_results, file_sym)
                    if representative_file == Symbol("")
                        representative_file = file_sym
                    end
                    push!(files_to_merge, file_sym)
                end
            end
            
            if representative_file == Symbol("")
                 println("Error: No data files found for $output_key category.")
                 return
            end

            unique!(files_to_merge)

            unified_header_parts = [output_key == :time_corr ? "t" : "r"]
            
            for file_sym in files_to_merge
                 header_str = get(FILE_HEADERS, String(file_sym), "")
                 parts = split(header_str, " ")[2:end] 
                 append!(unified_header_parts, parts)
            end
            
            println(io, join(unified_header_parts, " "))

            res_rep = file_results[representative_file]
            
            if res_rep["tmea"] === nothing || isempty(res_rep["tmea"])
                println("Error: Representative file $(representative_file) has no time/distance data.")
                return
            end

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
                        for _ in 1:num_cols_to_pad
                            push!(line_parts, NaN)
                        end
                    end
                end
                
                println(io, join([@sprintf("%.16f", x) for x in line_parts], " "))
            end
        end
        println("✅ Successfully wrote final file to $output_filename")
    end
end


# ==============================================================================
# --- Execution Flow (Unchanged) ---
# ==============================================================================

function main(L::Int, S::Int, T::Int, Tthermal::Int, run_ids::Vector{Int})
 
    time_corr_files = ["hh_tcorr.dat","hd_tcorr.dat","oo_tcorr.dat", "qq_tcorr.dat", "szsz_tcorr.dat", "aa_tcorr.dat", "bb_tcorr.dat", "cc_tcorr.dat", "dd_tcorr.dat"]
    equal_corr_files = ["hh_r_corr.dat", "oo_r_corr.dat", "szsz_r_corr.dat"]
    local_obs_files = ["h_expect.dat", "o_expect.dat", "q_expect.dat", "sz_expect.dat", "a_expect.dat", "b_expect.dat", "c_expect.dat", "d_expect.dat"]
    
    dynamic_files = String[]
    static_files = String[]
    dir = "data/L$(L)_S$(S)_T$(T)_Tthermal$(Tthermal)"
    all_run_dirs = glob(joinpath(dir, "run1"))
    for run_dir in all_run_dirs
        if isdir(run_dir)
            
            # 搜索 run 目录下所有 .dat 文件
            for filepath in glob(joinpath(run_dir, "*.dat"))
                filename = basename(filepath)
                
                # 排除已明确列出的文件和 profiles 文件
                if occursin("tcorr.dat", filename) && !(filename in time_corr_files)
                    if !(filename in dynamic_files)
                        push!(dynamic_files, filename)
                    end
                end
                if occursin("expect.dat", filename) && !(filename in local_obs_files)
                    if !(filename in static_files)
                        push!(static_files, filename)
                    end
                end
            end
        end
    end
    # Inside main function, after collecting dynamic_files:
    for filename in dynamic_files
        if !haskey(FILE_HEADERS, filename)
            # 为每个新发现的文件创建通用头部
            FILE_HEADERS[filename] = "t $(replace(filename, "_tcorr.dat" => "" ))_t_Avg $(replace(filename, "_tcorr.dat"  => "" ))_t_Err"
        end
    end
    for filename in static_files
        if !haskey(FILE_HEADERS, filename)
            # 为每个新发现的文件创建通用头部
            FILE_HEADERS[filename] = ["$(replace(filename, "_expect.dat" => "" ))_Avg"]
        end
    end
    # 将动态文件添加到主列表
    append!(time_corr_files, dynamic_files)
    append!(local_obs_files, static_files)
    @show time_corr_files
    time_results = process_file_group(L, S, T, Tthermal, run_ids, time_corr_files, :time_corr)
    equal_results = process_file_group(L, S, T, Tthermal, run_ids, equal_corr_files, :equal_corr)
    local_obs_results = process_file_group(L, S, T, Tthermal, run_ids, local_obs_files, :local_obs)
    
    dir = "data/L$(L)_S$(S)_T$(T)_Tthermal$(Tthermal)"
    
    write_local_output(dir, :time_corr, time_results)
    write_local_output(dir, :equal_corr, equal_results)
    write_local_output(dir, :local_obs, local_obs_results)
end


function process_all_folders(base_dir::String)
    dir_pattern = r"L(\d+)_S(\d+)_T(\d+)_Tthermal(\d+)"
    param_groups = Dict{NTuple{4, Int}, Vector{Int}}()
    
    for dir_path in glob(joinpath(base_dir, "*"))
        if isdir(dir_path)
            m = match(dir_pattern, basename(dir_path))
            if m !== nothing
                L = parse(Int, m.captures[1]); S = parse(Int, m.captures[2])
                T = parse(Int, m.captures[3]); Tthermal = parse(Int, m.captures[4])
                params = (L, S, T, Tthermal)

                run_pattern = r"run(\d+)"
                run_ids_for_params = Int[]
                for run_dir in glob(joinpath(dir_path, "run*"))
                    if isdir(run_dir)
                        run_m = match(run_pattern, basename(run_dir))
                        if run_m !== nothing
                            run_id = parse(Int, run_m.captures[1])
                            if !(run_id in run_ids_for_params)
                                push!(run_ids_for_params, run_id)
                            end
                        end
                    end
                end

                if haskey(param_groups, params)
                    append!(param_groups[params], run_ids_for_params)
                    unique!(param_groups[params])
                else
                    param_groups[params] = unique(run_ids_for_params)
                end
            end
        end
    end
    
    for (params, run_ids) in param_groups
        L, S, T, Tthermal = params
        if isempty(run_ids) continue end
        
        println("\n=========================================================")
        println("Processing parameter set: L=$L, S=$S, T=$T, Tthermal=$Tthermal")
        
        main(L, S, T, Tthermal, run_ids)
        
        println("=========================================================")
    end
    
    println("\nFinished processing all data.")
end


# --- Script Execution ---

process_all_folders("data")