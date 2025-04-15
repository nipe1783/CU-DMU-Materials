using POMDPs
using DMUStudent.HW6
using POMDPTools
using DiscreteValueIteration
using QuickPOMDPs
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using Plots
using Statistics: mean, std
using BasicPOMCP
using StatsBase


##################
# Problem 1: Tiger
##################

#--------
# Updater
#--------

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    m = up.m
    old_b_vec = b.b
    bp_vec = zeros(length(states(m)))

    # b'(s') ∝ P(o | a, s') * ∑_s [ T(s' | s, a) * b(s) ]
    for sp in states(m)
        sp_idx = stateindex(m, sp)

        # sum over all current states s
        trans_prob_sum = 0.0
        for s in states(m)
            s_idx = stateindex(m, s)
            trans_prob_sum += pdf(transition(m, s, a), sp) * old_b_vec[s_idx]
        end

        # multiply by observation probability
        obs_prob = pdf(observation(m, a, sp), o)
        bp_vec[sp_idx] = obs_prob * trans_prob_sum
    end

    # normalize
    norm_factor = sum(bp_vec)
    if norm_factor > 1e-12
        bp_vec ./= norm_factor
    else
        # fallback in case the observation is impossible
        fill!(bp_vec, 1.0 / length(bp_vec))
    end

    return DiscreteBelief(m, bp_vec)
end


# Note: you can access the transition and observation probabilities through the POMDPs.transtion and POMDPs.observation, and query individual probabilities with the pdf function. For example if you want to use more mathematical-looking functions, you could use the following:
# Z(o | a, s') can be programmed with
Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.

# This is needed to automatically turn any distribution into a discrete belief.
function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

# Note: to check your belief updater code, you can use POMDPTools: DiscreteUpdater. It should function exactly like your updater.

#-------
# Policy
#-------

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    b_vec = beliefvec(b)  # the belief vector in the correct stateindex order

    best_action_idx = 1
    best_value = -Inf

    for (i, alpha) in enumerate(p.alphas)
        # compute dot product of alpha and b_vec
        dot_val = 0.0
        for j in 1:length(alpha)
            dot_val += alpha[j] * b_vec[j]
        end

        # update best if this is higher
        if dot_val > best_value
            best_value = dot_val
            best_action_idx = i
        end
    end

    return p.alpha_actions[best_action_idx]
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order

#------
# QMDP
#------

function qmdp_solve(m; discount_factor=discount(m), max_iter=1000, tol=1e-6)
    svec = ordered_states(m)
    avec = ordered_actions(m)
    nS = length(svec)
    nA = length(avec)
    gamma = discount_factor  # now gamma is the numeric discount

    # ------------------------
    # 1) Initialize alpha vectors (k=0)
    # ------------------------
    alpha_vectors = [zeros(nS) for _ in 1:nA]

    # ------------------------
    # 2) Iteration
    # ------------------------
    for iter in 1:max_iter
        alpha_vectors_new = [copy(alpha_vectors[a]) for a in 1:nA]

        for (a_idx, a) in enumerate(avec)
            for (si, s) in enumerate(svec)
                r_sa = reward(m, s, a)

                sum_val = 0.0
                for (spi, sp) in enumerate(svec)
                    t_prob = pdf(transition(m, s, a), sp)
                    if t_prob > 0
                        best_alpha_sp = -Inf
                        for a2_idx in 1:nA
                            best_alpha_sp = max(best_alpha_sp, alpha_vectors[a2_idx][spi])
                        end
                        sum_val += t_prob * best_alpha_sp
                    end
                end

                alpha_vectors_new[a_idx][si] = r_sa + gamma * sum_val
            end
        end

        # Check for convergence
        delta = 0.0
        for a_idx in 1:nA
            diff_vec = alpha_vectors_new[a_idx] .- alpha_vectors[a_idx]
            local_delta = maximum(abs.(diff_vec))
            delta = max(delta, local_delta)
        end

        alpha_vectors = alpha_vectors_new

        if delta < tol
            # converged
            break
        end
    end

    return HW6AlphaVectorPolicy(alpha_vectors, avec)
end



# m = TigerPOMDP()

# qmdp_p = qmdp_solve(m)
# # Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
# sarsop_p = solve(SARSOPSolver(), m)
# up = HW6Updater(m)


# N = 5000
# max_steps = 500
# qmdp_returns = [
#     simulate(RolloutSimulator(max_steps=max_steps), m, qmdp_p, up)
#     for _ in 1:N
# ]
# mean_qmdp = mean(qmdp_returns)
# std_qmdp = std(qmdp_returns)
# sem_qmdp = std_qmdp / sqrt(N)
# println("QMDP Policy:")
# println("  Mean: ", mean_qmdp)
# println("  SEM:  ", sem_qmdp)

# sarsop_returns = [
#     simulate(RolloutSimulator(max_steps=max_steps), m, sarsop_p, up)
#     for _ in 1:N
# ]
# mean_sarsop = mean(sarsop_returns)
# std_sarsop = std(sarsop_returns)
# sem_sarsop = std_sarsop / sqrt(N)
# println("SARSOP Policy:")
# println("  Mean: ", mean_sarsop)
# println("  SEM:  ", sem_sarsop)


# qmdp_alphas = qmdp_p.alphas
# qmdp_actions = qmdp_p.alpha_actions
# sarsop_alphas = alphavectors(sarsop_p)

# function plot_alpha_vectors(alpha_vectors, labels; title_str="Alpha Vectors")
#     p = plot(title=title_str, legend=false)
#     for (i, alpha) in enumerate(alpha_vectors)
#         plot!(p, 0:1, alpha, label=labels[i], marker=:auto)
#     end
#     return p
# end

# # Plot QMDP
# qmdp_labels = string.(qmdp_actions)
# p_qmdp = plot_alpha_vectors(qmdp_alphas, qmdp_labels; title_str="QMDP Alpha Vectors")
# savefig(p_qmdp, "qmdp_alpha_vectors.png")

# # Plot SARSOP
# sarsop_labels = ["alpha$i" for i in 1:length(sarsop_alphas)]
# p_sarsop = plot_alpha_vectors(sarsop_alphas, sarsop_labels; title_str="SARSOP Alpha Vectors")
# savefig(p_sarsop, "sarsop_alpha_vectors.png")

##################
# Problem 2: Cancer
##################

cancerPOMDP = QuickPOMDP(
    states=[:healthy, :in_situ, :invasive, :death],
    actions=[:wait, :test, :treat],
    observations=[true, false],
    transition=function (s, a)
        if s == :healthy
            return SparseCat([:healthy, :in_situ], [0.98, 0.02])
        elseif s == :in_situ
            if a == :treat
                return SparseCat([:healthy, :in_situ], [0.6, 0.4])
            else
                return SparseCat([:in_situ, :invasive], [0.9, 0.1])
            end
        elseif s == :invasive
            if a == :treat
                return SparseCat([:healthy, :death, :invasive], [0.2, 0.2, 0.6])
            else
                return SparseCat([:invasive, :death], [0.4, 0.6])
            end
        else
            return Deterministic(:death)
        end
    end,
    observation=function (a, sp)
        if a == :test
            if sp == :healthy
                return SparseCat([true, false], [0.05, 0.95])
            elseif sp == :in_situ
                return SparseCat([true, false], [0.8, 0.2])
            elseif sp == :invasive
                return Deterministic(true)
            end
        elseif a == :treat
            if sp in (:in_situ, :invasive)
                return Deterministic(true)
            end
        end
        return Deterministic(false)
    end,
    reward=function (s, a)
        if s == :death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        end
    end,
    discount=0.99,
    initialstate=Deterministic(:healthy),
    isterminal=s -> s == :death,
)


@assert has_consistent_distributions(cancerPOMDP)

m = cancerPOMDP
up = HW6Updater(m)
qmdp_p = qmdp_solve(m)
sarsop_p = solve(SARSOPSolver(), m)
N = 1000
max_steps = 1000
qmdp_returns = [
    simulate(RolloutSimulator(max_steps=max_steps), m, qmdp_p, up)
    for _ in 1:N
]
mean_qmdp = mean(qmdp_returns)
std_qmdp = std(qmdp_returns)
sem_qmdp = std_qmdp / sqrt(N)
println("QMDP Policy:")
println("  Mean: ", mean_qmdp)
println("  SEM:  ", sem_qmdp)

sarsop_returns = [
    simulate(RolloutSimulator(max_steps=max_steps), m, sarsop_p, up)
    for _ in 1:N
]
mean_sarsop = mean(sarsop_returns)
std_sarsop = std(sarsop_returns)
sem_sarsop = std_sarsop / sqrt(N)
println("SARSOP Policy:")
println("  Mean: ", mean_sarsop)
println("  SEM:  ", sem_sarsop)

#####################
# Heuristic Policy
#####################

struct CancerHeuristicPolicy{QP<:Policy} <: Policy
    qmdp_p::QP
end

function POMDPs.action(p::CancerHeuristicPolicy, b::DiscreteBelief)
    b_vec = beliefvec(b)
    svec = ordered_states(b.pomdp)
    idx_healthy = findfirst(==(:healthy), svec)
    idx_in_situ = findfirst(==(:in_situ), svec)
    idx_invasive = findfirst(==(:invasive), svec)

    p_healthy = b_vec[idx_healthy]
    p_in_situ = b_vec[idx_in_situ]
    p_invasive = b_vec[idx_invasive]


    if p_healthy < 0.95 && p_healthy > 0.8
        return :test
    else
        return POMDPs.action(p.qmdp_p, b)
    end
end

heur_policy = CancerHeuristicPolicy(qmdp_p)
heur_returns = [
    simulate(RolloutSimulator(max_steps=max_steps), m, heur_policy, up)
    for _ in 1:N
]
mean_heur = mean(heur_returns)
std_heur = std(heur_returns)
sem_heur = std_heur / sqrt(N)
println("Heuristic Policy:")
println("  Mean: ", mean_heur)
println("  SEM:  ", sem_heur)

##################
# Problem 2: LaserTag
##################


function pomcp_solve(m)
    mdp = UnderlyingMDP(m)
    mdpSolver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=false)
    mdpPolicy = solve(mdpSolver, mdp)

    solver = POMCPSolver(
        tree_queries=100,
        max_depth=10,
        c=50.0,
        default_action=:measure,
        estimate_value=FORollout(mdpPolicy)
    )
    return solve(solver, m)
end


m = LaserTagPOMDP()


pomcp_p = pomcp_solve(m)
up = DiscreteUpdater(m)
println("Evaluating POMCP policy...")
@show HW6.evaluate((pomcp_p, up), n_episodes=1000)