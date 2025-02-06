using DMUStudent.HW2
using POMDPs: states, actions, transition, stateindex, actionindex, reward, pdf
using POMDPTools: ordered_states, render, VectorPolicy
using ElectronDisplay
using POMDPModels: SimpleGridWorld
using Distributions: support
using SparseArrays
using LinearAlgebra



# ############
# # Question 3
# ############

function lookahead(mdp, U, s, a, R, gamma, s_idx)
    # Computes Q(s, a) using the current value function U
    q_sa = R[a][s_idx]  # Immediate reward

    sp_dist = transition(mdp, s, a)
    for sp in support(sp_dist)
        p = pdf(sp_dist, sp)  # Transition probability from s to sp
        if p > 0
            sp_idx = stateindex(mdp, sp)
            q_sa += gamma * p * U[sp_idx]
        end
    end
    return q_sa
end

function backup(mdp, U, s, R, gamma)
    # Computes new U[s] pg. 142 in text. https://algorithmsbook.com/files/dm.pdf
    s_idx = stateindex(mdp, s)
    return maximum(lookahead(mdp, U, s, a, R, gamma, s_idx) for a in actions(mdp))
end

function value_iteration_viz(mdp, tol=1e-6, max_itrs=100, gamma=0.95)
    R = reward_vectors(mdp)
    U = zeros(length(states(mdp)))
    for itr = 1:max_itrs
        U = [backup(mdp, U, s, R, gamma) for s in states(mdp)]
        println("Iteration $itr")
        display(render(grid_world, color=U))
        sleep(0.2)
    end
    return U
end

function value_iteration(mdp, tol=1e-6, max_itrs=100, gamma=0.95)
    R = reward_vectors(mdp)
    U = zeros(length(states(mdp)))
    Unew = zeros(length(states(mdp)))
    for itr = 1:max_itrs
        Unew = [backup(mdp, U, s, R, gamma) for s in states(mdp)]
        residual = norm(Unew - U, Inf)
        println("Iteration $itr, residual: $residual")
        if norm(Unew - U, Inf) < tol
            break
        end
        U = copy(Unew)
    end
    return U
end


mdp = SimpleGridWorld()
U = value_iteration(mdp)

############
# Question 4
############

mdp = UnresponsiveACASMDP(7)
U = value_iteration(mdp, 1e-6, 100, 0.99)
@show HW2.evaluate(U, "nicolas.perrault@colorado.edu")
