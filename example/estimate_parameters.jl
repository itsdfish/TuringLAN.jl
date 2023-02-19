###################################################################################################
#                                        Load Packages
###################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../")
using Plots
using Flux
using Distributions
using Random
using Turing
using Distributions
using ReverseDiff 
using Zygote

using BSON: @load
import Distributions: ContinuousUnivariateDistribution
import Distributions: logpdf, loglikelihood
Random.seed!(58420)
@load "gaussian_model.bson" model
###################################################################################################
#                                     estimate parameters
###################################################################################################
struct VSModel{T,T1} <: ContinuousUnivariateDistribution
    μ::T
    neural_net::T1
end

loglikelihood(d::VSModel, data::Vector{<:Real}) = logpdf(d, data)

# function logpdf(m::VSModel, data::Vector{<:Real})
#     μ = m.μ
#     model = m.neural_net
#     σ = 1.0
#     LLs = map(x -> model([μ,σ,x])[1], data)
#     return sum(LLs)
# end

function logpdf(m::VSModel, data::Vector{<:Real})
    n = length(data)
    μ = m.μ
    model = m.neural_net
    σ = 1.0
    fill(μ, 1, n)
    m_data = [fill(μ, 1, n);
                    fill(σ, 1, n);
                    data']
    LLs = model(m_data)
    return sum(LLs)
end

@model function my_model(model, data)
    μ ~ Uniform(-3, 3)
    data ~ VSModel(μ, model)
end

data = Float32.(rand(Normal(0, 1), 150))

# Turing.setadbackend(:forwarddiff)
# Turing.setadbackend(:reversediff)
Turing.setadbackend(:zygote)

delta = 0.65
n_adapt = 1000
specs = NUTS(n_adapt, delta)
n_samples = 1000
chain = sample(my_model(model, data), specs, n_samples)