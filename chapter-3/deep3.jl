⋅ = *

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

    a1 = x ⋅ W1 + b1
    z1 = h(a1)
    a2 = z1 ⋅ W2 + b2
    z2 = h(a2)
    a3 = z2 ⋅ W3 + b3
    σ(a3)
end

network = init_network()
x = [1.0 0.5]
forward(network, x)