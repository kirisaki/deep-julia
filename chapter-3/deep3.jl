using LinearAlgebra
using MLDatasets.MNIST
using ImageCore
using Plots
using Random
using PyCall
@pyimport pickle

function h(v)
    map(x -> 1.0 / (1.0 + exp(-x)), v)
end

function σ(a)
    c = maximum(a)
    exp_a = map(x -> exp(x .- c), a)
    exp_a / sum(exp_a)
end

function init_network()::Matrix{Float64}
    rand(Float64, (2,3))
end

function loss(
    x::Matrix{Float64},
    t::Matrix{Float64},
    W::Matrix{Float64},
    )::Float64
    z = x * W
    y = σ(z)
    cross_entropy_error(y, t)
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

function prepare_testdata(;batch=1)
    data, labels = MNIST.testdata()
    batch_num = div(length(labels), batch)
    data0 = eachslice(reshape(data, (28 * 28 * batch, batch_num)), dims=2)
    labels0 = eachslice(transpose(reshape(labels, (batch, batch_num))), dims=1)
    collect(zip(data0, labels0))
end

function cross_entropy_error(
    y::AbstractArray{T, N}, 
    t::AbstractArray{T, N},
    )::AbstractFloat where{T <: Number, N}
    - sum(float.(t) .* log.(float.(y)))
end


function numerical_gradient(
    f::Function,
    xs::AbstractArray{T, N};
    h=1e-4::T,
    )::AbstractArray{T, N} where{T <: Number, N}
    grad = zeros(size(xs))
    for (i, x) in enumerate(xs)
        xs[i] = x + h
        fxh1 = f(xs)
        xs[i] = x - h
        fxh2 = f(xs)
        grad[i] = (fxh1 - fxh2) / (2h)
        xs[i] = x
    end
    grad
end

function gradient_descent(
    f::Function,
    init_x::AbstractArray{T, N};
    learning_rate=0.01::T,
    steps=100::UInt128
    )::AbstractArray{T, N} where{T <: Number, N}
    x = init_x
    for _ = 1:steps
        x -= learning_rate * numerical_gradient(f, x)
    end
    x
end

function make_predict(fname="./sample_weight.pkl")
    f = pybuiltin("open")(fname, "rb")
    network = pickle.load(f)
    f.close()
    W₁ ,W₂, W₃ = network["W1"], network["W2"], network["W3"]
    b₁ ,b₂, b₃ = network["b1"], network["b2"], network["b3"]
    function (x::AbstractArray)
        a₁ = x * W₁ .+ transpose(b₁)
        z₁ = h(a₁)
        a₂ = z₁ * W₂ .+ transpose(b₂)
        z₂ = h(a₂)
        a₃ = z₂ * W₃ .+ transpose(b₃)
        σ(a₃)
    end
end

function bench01()
    predict = make_predict()
    data = prepare_testdata()
    count = 0
    for (x, l) in data
        if argmax(predict(x))[2] - 1 == l[1]
            count += 1
        end
    end
    count / length(data)
end
bench01()

function bench01_batch(;batch=100)
    predict = make_predict()
    data = prepare_testdata(batch=batch)
    count = 0
    for (xs, ls) in data
        for (p, l) in zip(eachslice(predict(transpose(reshape(xs, (28 * 28, batch)))), dims=1), ls)
            if argmax(p) - 1 == l
                count += 1
            end
        end
    end
    count / (length(data) * batch)
end