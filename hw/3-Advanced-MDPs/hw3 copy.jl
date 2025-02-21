using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics
using BenchmarkTools: @btime
using POMDPTools
using Distributions: support


## Questions 2:

mdp = HW3.DenseGridWorld(seed=3)

# Questions 2.a:
function rollout(mdp, policy_function, s0, max_steps=100)

    r_total = 0.0
    t = 0
    s = s0

    while !(s in mdp.terminate_from) && t < max_steps
        a = policy_function(mdp, s)
        while true
            a = policy_function(mdp, s)
            a1 = policy_function(mdp, s)
            if a == a1
                break
            end
        end
        #println("choosing action: ", a)
        s, r = @gen(:sp, :r)(mdp, s, a)
        r_total += r
        t += 1
    end

    return r_total
end

function random_policy(mdp, s)
    return rand(actions(mdp))
end


# results = [rollout(mdp, random_policy, rand(initialstate(mdp))) for _ in 1:5000]
# @show random_p_mean_results = mean(results)
# @show random_p_SEM_results = std(results) / sqrt(length(results))

# Question 2.b

function heuristic_policy(mdp, s)
    max_a = nothing
    max_r = -Inf
    for a in actions(mdp)
        sp, r = @gen(:sp, :r)(mdp, s, a)
        _, r = @gen(:sp, :r)(mdp, sp, a)
        #println("State: ", s, " Action: ", a, " Reward: ", r)
        if r > max_r
            max_r = r
            max_a = a
        end
    end

    return max_a
end

# results = [rollout(mdp, heuristic_policy, rand(initialstate(mdp))) for _ in 1:5000]
# @show heuristic_p_mean_results = mean(results)
# @show heuristic_p_SEM_results = std(results) / sqrt(length(results))

# Question 3:  

struct MonteCarloTreeSearch
    mdp::HW3.DenseGridWorld
    N::Dict{Tuple{Any,Any},Int}
    Q::Dict{Tuple{Any,Any},Float64}
    T::Dict{Tuple{Any,Any,Any},Int}
    d::Int
    m::Int
    c::Float64
end

function MonteCarloTreeSearch(mdp::HW3.DenseGridWorld, d::Int=100, m::Int=7, c::Float64=200.0)
    S = statetype(mdp)
    A = actiontype(mdp)
    N = Dict{Tuple{S,A},Int}()
    Q = Dict{Tuple{S,A},Float64}()
    T = Dict{Tuple{S,A,S},Int}()
    return MonteCarloTreeSearch(mdp, N, Q, T, d, m, c)
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns) / Nsa)
function explore(mcts::MonteCarloTreeSearch, s::Any)
    Ns = sum([mcts.N[(s, a)] for a in actions(mcts.mdp)])
    #println("Ns: ", Ns)

    for a in actions(mcts.mdp)
        val = mcts.Q[(s, a)] + mcts.c * bonus(mcts.N[(s, a)], Ns)
        #println("Action: ", a, " Value: ", val)
    end


    a_ind = argmax([mcts.Q[(s, a)] + mcts.c * bonus(mcts.N[(s, a)], Ns) for a in actions(mcts.mdp)])
    return actions(mcts.mdp)[a_ind]
end

function simulate!(mcts::MonteCarloTreeSearch, s::Any, d::Int)
    #println("State: ", s)

    if d <= 0
        val = rollout(mcts.mdp, heuristic_policy, s)
        return val
    end

    if !haskey(mcts.N, (s, actions(mcts.mdp)[1]))
        #println("Initializing state: ", s)
        for a in actions(mcts.mdp)
            mcts.N[(s, a)] = 0
            mcts.Q[(s, a)] = 0.0
        end
        #println("Initialized state: ", s)
        val = rollout(mcts.mdp, heuristic_policy, s)
        #println("Rolled out state: ", s, " Value: ", val)
        return val
    end

    #println("Exploring state: ", s)
    a = explore(mcts, s)
    #println("Explored state: ", s, " Action: ", a)
    sp, r = @gen(:sp, :r)(mcts.mdp, s, a)
    # #println("state: ", s, " Action: ", a, " Next State: ", sp, " Reward: ", r)
    q = r + discount(mcts.mdp) * simulate!(mcts, sp, d - 1)
    # #println("state: ", s, " Value: ", q)
    mcts.N[(s, a)] += 1
    if !haskey(mcts.T, (s, a, sp))
        mcts.T[(s, a, sp)] = 0
    end
    mcts.T[(s, a, sp)] += 1
    mcts.Q[(s, a)] += (q - mcts.Q[(s, a)]) / mcts.N[(s, a)]
    return q
end

function (mcts::MonteCarloTreeSearch)(s::Any)
    for k in 1:mcts.m
        # #println("Iteration: ", k)
        simulate!(mcts, s, mcts.d)
        inchrome(visualize_tree(mcts.Q, mcts.N, mcts.T, SA[19, 19]))
    end
end

# mdp = HW3.DenseGridWorld(seed=4)
# mcts = MonteCarloTreeSearch(mdp)
# mcts(SA[19, 19])

# Question 4:
function select_action(mdp, s)

    mcts = MonteCarloTreeSearch(mdp)
    mcts(s)
    a = argmax([mcts.Q[(s, a)] for a in actions(mdp)])
    return actions(mdp)[a]

end

# R = []
# for i in 1:100

#     s = rand(initialstate(mdp)) # random init state
#     r_sum = 0.0
#     for step in 1:100
#         a = select_action(mdp, s)
#         s, r = @gen(:sp, :r)(mdp, s, a)
#         r_sum += r
#         @show r_sum
#         if s in mdp.terminate_from
#             break
#         end
#     end
#     push!(R, r_sum)
# end

# mean_r = mean(R)
# std_r = std(R)
# SEM_r = std_r / sqrt(length(R))
# @show mean_r
# @show std_r
# @show SEM_r

# Question 5:
HW3.evaluate(select_action, "nicolas.perrault@colorado.edu")