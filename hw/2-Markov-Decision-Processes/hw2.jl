import Pkg
Pkg.activate(".")  # Ensure you're in the correct environment
Pkg.instantiate()  # Install missing dependencies
Pkg.status()       # Check that Distributions.jl is installed


using DMUStudent.HW2
using POMDPs: states, actions, transition, stateindex, actionindex, reward
using POMDPTools: ordered_states, render, VectorPolicy
using ElectronDisplay
using POMDPModels: SimpleGridWorld
using Distributions: support


############
# Question 3
############

function lookahead(mdp, U, s, a, T, R, gamma, s_idx)
    # Computes Q(s, a) using the current value function U
    q_sa = R[a][s_idx]  # Immediate reward

    for sp in support(transition(mdp, s, a))
        sp_idx = stateindex(mdp, sp)
        p = T[a][s_idx, sp_idx]  # Transition probability from s to sp
        q_sa += gamma * p * U[sp_idx]
    end

    return q_sa
end

function backup(mdp, U, s, T, R, gamma)
    # Computes new U[s] pg. 142 in text. https://algorithmsbook.com/files/dm.pdf
    s_idx = stateindex(mdp, s)
    return maximum(lookahead(mdp, U, s, a, T, R, gamma, s_idx) for a in actions(mdp))
end

function value_iteration_viz(mdp, tol=1e-6, max_itrs=100, gamma=0.95)
    T = transition_matrices(mdp)
    R = reward_vectors(mdp)
    U = zeros(length(states(mdp)))
    A = collect(actions(mdp))
    for itr = 1:max_itrs
        U = [backup(mdp, U, s, T, R, gamma) for s in states(mdp)]
        display(render(grid_world, color=U))
        sleep(0.2)
    end
    return U
end

function value_iteration(mdp, tol=1e-6, max_itrs=100, gamma=0.95)
    T = transition_matrices(mdp)
    R = reward_vectors(mdp)
    U = zeros(length(states(mdp)))
    for itr = 1:max_itrs
        U = [backup(mdp, U, s, T, R, gamma) for s in states(mdp)]
    end
    return U
end


mdp = SimpleGridWorld()
U = value_iteration_viz(mdp)
