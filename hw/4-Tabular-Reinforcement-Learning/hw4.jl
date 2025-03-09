using DMUStudent.HW4
using POMDPs
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics
using BenchmarkTools: @btime
using POMDPTools
using Distributions: support
using CommonRLInterface: render, actions, act!, observe, reset!, AbstractEnv, observations, terminated, clone
using Plots
using POMDPModels: SimpleGridWorld



# Sarsa:
function sarsa_episode!(Q, env; epsilon=0.10, gamma=1.0, alpha=0.1)
    start = time()

    function policy(s)
        if rand() < epsilon
            return rand(actions(env))
        else
            return argmax(a -> Q[(s, a)], actions(env))
        end
    end

    s = observe(env)
    a = policy(s)
    r = act!(env, a)
    sp = observe(env)
    hist = [s]

    while !terminated(env)
        ap = policy(sp)

        Q[(s, a)] += alpha * (r + gamma * Q[(sp, ap)] - Q[(s, a)])

        s = sp
        a = ap
        r = act!(env, a)
        sp = observe(env)
        push!(hist, sp)
    end

    Q[(s, a)] += alpha * (r - Q[(s, a)])

    return (hist=hist, Q=copy(Q), time=time() - start)
end

function sarsa!(env; n_episodes=100)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []

    for i in 1:n_episodes
        reset!(env)
        push!(episodes, sarsa_episode!(Q, env;
            epsilon=max(0.1, 1 - i / n_episodes)))
    end

    return episodes
end

# Q Learning:
function q_learning_episode!(Q, env; epsilon=0.10, gamma=1.0, alpha=0.2)
    start = time()

    function policy(s)
        if rand() < epsilon
            return rand(actions(env))
        else
            return argmax(a -> Q[(s, a)], actions(env))
        end
    end

    s = observe(env)
    hist = [s]

    while !terminated(env)
        a = policy(s)
        r = act!(env, a)
        sp = observe(env)
        Q[(s, a)] += alpha * (r + gamma * maximum(Q[(sp, ap)] for ap in actions(env)) - Q[(s, a)])
        s = sp
        push!(hist, s)
    end

    return (hist=hist, Q=copy(Q), time=time() - start)
end

function q_learning!(env; n_episodes=100)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []

    for i in 1:n_episodes
        reset!(env)
        push!(episodes, q_learning_episode!(Q, env;
            epsilon=max(0.1, 1 - i / n_episodes)))
    end

    return episodes
end

mdp = GridWorldEnv()
env = mdp
# episodes = Dict("SARSA" => sarsa!(env, n_episodes=200000), "Q-Learning" => q_learning!(env, n_episodes=200000))
episodes = Dict("Q-Learning" => q_learning!(env, n_episodes=200000))
# episodes = Dict("SARSA" => sarsa!(env, n_episodes=200000))


function evaluate(env, policy, n_episodes=200000, gamma=1.0)
    returns = Float64[]
    for _ in 1:n_episodes
        t = 0
        r = 0.0
        reset!(env)
        s = observe(env)
        while !terminated(env)
            a = policy(s)
            r += gamma^t * act!(env, a)
            s = observe(env)
            t += 1
        end
        push!(returns, r)
    end
    return returns
end

p = plot(xlabel="steps in environment", ylabel="avg return")
n = 20000
stop = 200000
for (name, eps) in episodes
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    xs = [0]
    ys = [mean(evaluate(env, s -> argmax(a -> Q[(s, a)], actions(env))))]
    for i in n:n:min(stop, length(eps))
        newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
        push!(xs, last(xs) + newsteps)
        Q = eps[i].Q
        push!(ys, mean(evaluate(env, s -> argmax(a -> Q[(s, a)], actions(env)))))
    end
    plot!(p, xs, ys, label=name)
end
p
savefig(p, "plot1.png")


p = plot(xlabel="wall clock time", ylabel="avg return")
n = 20000
stop = 200000
for (name, eps) in episodes
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    xs = [0.0]
    ys = [mean(evaluate(env, s -> argmax(a -> Q[(s, a)], actions(env))))]
    for i in n:n:min(stop, length(eps))
        newtime = sum(ep.time for ep in eps[i-n+1:i])
        push!(xs, last(xs) + newtime)
        Q = eps[i].Q
        push!(ys, mean(evaluate(env, s -> argmax(a -> Q[(s, a)], actions(env)))))
    end
    plot!(p, xs, ys, label=name)
end
p
savefig(p, "plot2.png")