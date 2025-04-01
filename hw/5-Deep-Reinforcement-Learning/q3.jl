# ############
# # Question 3
# ############

using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs
using Plots
using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper


function epsilon_greedy_action(env, Q, s, eps)
    if rand() < eps
        return rand(1:length(actions(env)))
    else
        return argmax(Q(s))
    end
end

function loss(Q, Q_target, s, a_ind, r, sp, terminal; gamma=0.99)
    if terminal
        return (r - Q(s)[a_ind])^2
    end
    return (r + gamma * maximum(Q_target(sp)[ap_ind] for ap_ind in 1:length(actions(env))) - Q(s)[a_ind])^2
end

function dqn(env; maxBufferSize=100000, batchSize=32, updateFrequency=1000)

    Q = Chain(Dense(2, 128, relu),
        Dense(128, length(actions(env))))
    Q_target = deepcopy(Q)
    opt = Flux.setup(ADAM(0.0005), Q)

    theBuffer = []
    lossexperience = Float32[] # for plotting
    rewardexperience = Float32[] # for plotting

    # Initialize state, terminal flag, and number of steps
    s = observe(env)
    terminal = false
    stepNum = 0
    training = true
    maxSteps = 150000

    while training && stepNum < maxSteps

        # take epsilon greedy action, build experience, store in theBuffer
        eps = max(0.05, 0.5 - stepNum / maxBufferSize)
        a_ind = epsilon_greedy_action(env, Q, s, eps)
        r = act!(env, actions(env)[a_ind])
        sp = observe(env)
        terminal = terminated(env)
        stepNum += 1
        experience_tuple = (s, a_ind, r, sp, terminal)
        push!(theBuffer, experience_tuple)

        # remove oldest experience from buffer
        if length(theBuffer) > maxBufferSize
            popfirst!(theBuffer)
        end

        # Sample a batch of experiences from theBuffer
        avgLoss = 0.0
        for experience in rand(theBuffer, batchSize)
            lossVal, grads = Flux.withgradient(loss, Q, Q_target, experience...)
            Flux.update!(opt, Q, grads[1])
            avgLoss += lossVal
        end

        if terminal
            reset!(env)
        else
            s = sp
        end

        # Update the target network every updateFrequency steps
        if stepNum % updateFrequency == 0
            Q_target = deepcopy(Q)
        end

        # Log the average loss and reward every 1000 steps
        avgReward = 0.0
        if stepNum % 1000 == 0
            eval = HW5.evaluate(s -> actions(env)[argmax(Q(s[1:2]))], n_episodes=1000)
            avgReward = eval[:score]
            println("Steps: $stepNum, Avg Reward: $avgReward")
            push!(lossexperience, avgLoss / batchSize)
            push!(rewardexperience, avgReward)
        end

        # Check if training should stop
        if avgReward >= 40.0
            eval = HW5.evaluate(s -> actions(env)[argmax(Q(s[1:2]))], "nicolas.perrault@colorado.edu", n_episodes=10000)
            if eval[:score] >= 40.0
                training = false
            end
        end
    end

    return Q, lossexperience, rewardexperience
end

env = QuickWrapper(HW5.mc,
    actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
    observe=mc -> observe(mc)[1:2]
)

result = dqn(env)
Q = result[1] # Extract the trained Q network from the result
loss_curve = result[2] # Extract the loss curve for plotting
reward_curve = result[3] # Extract the reward curve for plotting
number_steps = length(loss_curve)

#----------
# Rendering
#----------

# You can show an image of the environment like this (use ElectronDisplay if running from REPL):
display(render(env))

# The following code allows you to render the value function
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value")
savefig("DQN_ValueFunc.png")

# Plot the loss and reward curves over training iterations
p1 = plot(loss_curve; label="Loss Vs Iterations", xlabel="Training Steps (x 1000)", ylabel="Loss", title="Loss Vs Iterations", yaxis=:log)
plot(p1; legend=:topright)
savefig("DQN_loss_curve.png") # Save the plot to a file

p2 = plot(reward_curve; label="Reward Curve", xlabel="Training Steps (x 1000)", ylabel="Average Reward", title="Average Reward Vs Iterations") # Plot the reward curve 
plot(p2; legend=:topright)
savefig("DQN_reward_curve.png") # Save the plot to a file