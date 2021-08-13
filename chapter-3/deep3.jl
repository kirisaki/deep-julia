using LinearAlgebra
using MLDatasets.MNIST
using ImageCore
using Plots
using StatsBase

function h(v)
    map(x -> 1.0 / (1.0 + exp(-x)), v)
end

function σ(a)
    c = maximum(a)
    exp_a = map(x -> exp(x .- c), a)
    exp_a / sum(exp_a)
end

function init_network()
    network = Dict()
    network["W1"] = [0.1 0.3 0.5; 0.2 0.4 0.6]
    network["b1"] = [0.1 0.2 0.3]
    network["W2"] = [0.1 0.4; 0.2 0.5; 0.3 0.6]
    network["b2"] = [0.1 0.2]
    network["W3"] = [0.1 0.3; 0.2 0.4]
    network["b3"] = [0.1 0.2]

    return network
end

function forward(network, x)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = x * W1 + b1
    z1 = h(a1)
    a2 = z1 * W2 + b2
    z2 = h(a2)
    a3 = z2 * W3 + b3
end

network = init_network()
x = [1.0 0.5]
forward(network, x)
data, labels = MNIST.testdata(UInt8)
datasets = collect(zip(eachslice(data, dims=3), labels))
datasets0 = collect(zip(map(arr -> collect(reshape(arr, (28 * 28))), eachslice(data, dims=3)), labels))
StatsBase.sample(datasets, 10)
d, _ =first(datasets0)
size(d)
plot(MNIST.convert2image(d))

function cross_entropy_error(y::AbstractArray{T, N}, t::AbstractArray{T, N})::AbstractFloat where{T <: Number, N}
    - sum(float.(t) .* log.(float.(y))) / length(t)
end

cross_entropy_error(d, d)

function numerical_gradient(f::Function, xs::AbstractVector{T}; h=1e-6)::AbstractVector{T} where{T <: Number}
    grad = zeros(size(xs))
    for (i, x) in enumerate(xs)
        xs0 = xs
        tmp_val = xs0[i]
        xs0[i] = tmp_val + h
        fxh1 = f(xs0)
        xs0[i] = tmp_val - h
        fxh2 = f(xs0)
        grad[i] = (fxh1 - fxh2) / (2h)
    end
    grad
end

f(xs) = xs[1]^2 + xs[2]^2

numerical_gradient(f, [.0, .0])

function gradient_descent(f::Function, init_x::AbstractVector; learning_rate=0.01, steps=100)
    x = init_x
    for _ = 1:steps
        x -= learning_rate .* numerical_gradient(f, x)
    end
    x
end
gradient_descent(f, [-3.0, 4.0], learning_rate=.001, steps=100)