# ############
# # Question 1
# ############

using DMUStudent.HW5: HW5, mc
using QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic, Uniform, SparseCat, FunctionPolicy, RolloutSimulator
using Statistics: mean
import POMDPs


cancer_monitoring = QuickPOMDP(
    states=[:healthy, :in_situ_cancer, :invasive_cancer, :death],
    actions=[:wait, :test, :treat, :wait, :test, :treat],
    observations=[:healthy, :in_situ_cancer],

    # transition should be a function that takes in s and a and returns the distribution of s'
    transition=function (s, a)
        if s == :healthy
            return SparseCat([:healthy, :in_situ_cancer], [0.98, 0.02])
        end
        if s == :in_situ_cancer
            if a == :treat
                return SparseCat([:healthy, :in_situ_cancer, :invasive_cancer], [0.6, 0.2, 0.2])
            else
                return SparseCat([:in_situ_cancer, :invasive_cancer], [0.9, 0.1])
            end
        end
        if s == :invasive_cancer
            if a == :treat
                return SparseCat([:healthy, :invasive_cancer, :death], [0.2, 0.6, 0.2])
            else
                return SparseCat([:invasive_cancer, :death], [0.4, 0.6])
            end
        else # s == :death
            return Deterministic(:death)
        end
    end,

    # observation should be a function that takes in s, a, and sp, and returns the distribution of o
    observation=function (s, a, sp)
        if a == :test
            if sp == :healthy
                return SparseCat([:positive, :negative], [0.05, 0.95])
            elseif sp == :in_situ_cancer
                return SparseCat([:positive, :negative], [0.8, 0.2])
            elseif sp == :invasive_cancer
                return SparseCat([:positive], [1.0])
            end
        else
            return SparseCat([:negative], [1.0])
        end
    end, reward=function (s, a)
        if s == :death
            return 0.0
        elseif a == :wait
            return 1.0
        elseif a == :test
            return 0.8
        elseif a == :treat
            return 0.1
        end
    end, initialstate=Uniform([:healthy, :in_situ_cancer]), discount=0.99
)

# evaluate with policy that always waits
policy = FunctionPolicy(o -> :wait)
sim = RolloutSimulator(max_steps=100)
@show @time mean(POMDPs.simulate(sim, cancer_monitoring, policy) for _ in 1:10_000)