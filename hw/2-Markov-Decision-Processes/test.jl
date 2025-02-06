using POMDPs
using POMDPModels: SimpleGridWorld
using POMDPTools: render,
    Policies.policy_transition_matrix,
    Policies.policy_reward_vector,
    ordered_states,
    VectorPolicy
using LinearAlgebra: I

m = SimpleGridWorld()
display(render(m))