using Random # 用于 shuffle, rand 等

# 为了代码可测试，我们先定义一个结构体
# mutable struct BracketSystem
#     L::Int
#     S::Int
#     BracketConfig::Vector{Int}
    
#     BracketSystem(L, S) = new(L, S, zeros(Int, L))
# end

"""
    generate_initial_state(L::Int, S::Int; method::String="random")

为Motzkin链生成一个保证合法的初始构型。

这个函数取代了原有的 InitialConfigSpinS!，因为它不依赖于一个已有的 sys 对象，
而是直接返回一个新的构型向量。

# Arguments
- `L::Int`: 链的长度。
- `S::Int`: 颜色的种类数 (S > 0)。
- `method::String`: 生成方法，可选值为 "zeros" 或 "random"。

# Returns
- `Vector{Int}`: 代表Motzkin路径的向量。`k` 代表 uᵏ, `-k` 代表 dᵏ, `0` 代表平坦。
"""
function generate_initial_state(L::Int, S::Int; method::String="random")
    if method == "zeros"
        # 方法1: 最简单的合法构型是全0路径
        return zeros(Int, L)
        
    elseif method == "random"
        # 方法2: 构造法，保证生成的路径总是合法的
        path = Int[]
        color_stack = Int[] # 用于确保障色括号嵌套

        for i in 1:L
            possible_moves = Int[]
            remaining_steps = L - (i - 1)

            # 检查是否必须关闭一个括号以确保路径在结尾闭合
            if length(color_stack) == remaining_steps
                # 必须下降来关闭栈中未匹配的 u
                top_color = pop!(color_stack)
                push!(path, -top_color)
                continue
            end

            # 1. 考虑走平步 (0)
            push!(possible_moves, 0)

            # 2. 考虑向上走 (uᵏ)
            # 条件：剩下的步数足够关闭所有已打开 + 将要打开的括号
            if length(color_stack) < remaining_steps - 1
                for k in 1:S
                    push!(possible_moves, k)
                end
            end

            # 3. 考虑向下走 (dᵏ)
            # 条件：栈中必须有未关闭的 u
            if !isempty(color_stack)
                top_color = last(color_stack)
                push!(possible_moves, -top_color)
            end
            
            # 从所有可能的合法移动中随机选择一个
            chosen_move = rand(possible_moves)
            push!(path, chosen_move)

            # 更新颜色栈
            if chosen_move > 0 # UP
                push!(color_stack, chosen_move)
            elseif chosen_move < 0 # DOWN
                pop!(color_stack)
            end
        end
        return path
        
    else
        error("无效的方法名。请选择 'zeros' 或 'random'。")
    end
end

# 辅助函数，用于美观地打印路径
function pretty_print_path(path::Vector{Int})
    repr_list = String[]
    for spin in path
        if spin == 0
            push!(repr_list, "0")
        elseif spin > 0
            push!(repr_list, "u$(spin)")
        else
            push!(repr_list, "d$(-spin)")
        end
    end
    println(join(repr_list, " "))
end

# --- 主函数：使用示例 ---
function main()
    L = 12 # 链长
    S = 2  # 颜色种类

    println("链长 L = $L, 颜色数 S = $S\n")

    # 1. 生成一个全0的初始状态
    println("方法 'zeros':")
    sys_zero = BracketSystem(L, S)
    sys_zero.BracketConfig = generate_initial_state(L, S, method="zeros")
    pretty_print_path(sys_zero.BracketConfig)
    println("-"^20)

    # 2. 生成几个随机的、保证合法的初始状态
    println("方法 'random' (生成5个例子):")
    for _ in 1:5
        sys_random = BracketSystem(L, S)
        sys_random.BracketConfig = generate_initial_state(L, S, method="random")
        # pretty_print_path(sys_random.BracketConfig)
        print(sys_random.BracketConfig)
    end
end

# 运行主函数
# main()