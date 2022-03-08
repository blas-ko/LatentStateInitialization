############################# DESCENT OPTIMIZERS ###############################
# Inspired by:
# https://github.com/denizyuret/Knet.jl/blob/master/src/update.jl

abstract type Optimiser; end

#GD
mutable struct GradientDescent <: Optimiser
    lr::AbstractFloat
end
GradientDescent(; lr=0.01) = GradientDescent(lr)


function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::GradientDescent; kwargs...)
    g, Jx = finite_differences_gradient(system, x0, data; kwargs...)

    x0 .-= p.lr*g
    return Jx
end

#Momentum
mutable struct Momentum <: Optimiser
    lr::AbstractFloat
    gamma::AbstractFloat
    velocity
end
Momentum(; lr=0.01, gamma=0.9, velocity=nothing) = Momentum(lr, gamma, velocity)

function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::Momentum; kwargs...)
    g, Jx = finite_differences_gradient(system, x0, data; kwargs...)

    p.velocity ===nothing && (p.velocity = zero(x0))
    p.velocity = p.gamma*p.velocity + p.lr*g

    x0 .-= p.velocity
    return Jx
end

#Nesterov
mutable struct Nesterov <: Optimiser
    lr::AbstractFloat
    gamma::AbstractFloat
    velocity
end
Nesterov(; lr=0.01, gamma=0.9, velocity=nothing) = Nesterov(lr, gamma, velocity)

function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::Nesterov;kwargs...)
    p.velocity ===nothing && (p.velocity = zero(x0))
    g, Jx = finite_differences_gradient(system, x0 - p.gamma * p.velocity, data; kwargs...)
    p.velocity = p.gamma*p.velocity + p.lr*g

    x0 .-= p.velocity
    return Jx
end

#Adagrad
mutable struct Adagrad <: Optimiser
    lr::AbstractFloat
    epsilon::AbstractFloat
    G
end
Adagrad(; lr=0.01, epsilon=sqrt(eps()), G=nothing) = Adagrad(lr, epsilon, G)

function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::Adagrad; kwargs...)
    g, Jx = finite_differences_gradient(system, x0, data; kwargs...)
    p.G ===nothing && (p.G = zero(x0))
    p.G += g .* g

    x0 .-= p.lr * (g ./sqrt.(p.G .+ p.epsilon))
    return Jx
end

#Adadelta
mutable struct Adadelta <: Optimiser
    lr::AbstractFloat
    epsilon::AbstractFloat
    rho::AbstractFloat
    delta
    G
end
Adadelta(; lr=1.0, epsilon=sqrt(eps()), rho=0.9, delta=nothing, G=nothing) = Adadelta(1.0, epsilon, rho, delta, G)

function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::Adadelta; kwargs...)
    g, Jx = finite_differences_gradient(system, x0, data; kwargs...)
    if p.G===nothing; p.G = zero(x0); p.delta = zero(x0); end
    p.G = p.rho * p.G .+ (1-p.rho) * g .* g
    dx0 = sqrt.( p.delta .+ p.epsilon )./ sqrt.( p.G .+ p.epsilon ) .* g
    p.delta = p.rho*p.delta .+ (1-p.rho)* dx0 .* dx0

    x0 .-= dx0
    return Jx
end

#Rmsprop
mutable struct Rmsprop <: Optimiser
    lr::AbstractFloat
    epsilon::AbstractFloat
    rho::AbstractFloat
    G
end
Rmsprop(; lr=0.01, epsilon=sqrt(eps()), rho=0.9, G=nothing) = Rmsprop(lr, epsilon, rho, G)

function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::Rmsprop; kwargs...)
    g, Jx = finite_differences_gradient(system, x0, data; kwargs...)
    p.G ===nothing && (p.G = zero(x0))
    p.G = p.rho * p.G .+ (1-p.rho) * g .* g

    x0 .-= p.lr * (g ./ sqrt.(p.G .+ p.epsilon))
    return Jx
end

#Adam
mutable struct Adam <: Optimiser
    lr::AbstractFloat
    epsilon::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    t::Int
    v
    m
end
Adam(; lr=0.01, epsilon=sqrt(eps()), beta1=0.9, beta2=0.99, t=0, v=nothing, m=nothing) = Adam(lr, epsilon, beta1, beta2, t, v, m)

function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::Adam; kwargs...)
    g, Jx = finite_differences_gradient(system, x0, data; kwargs...)
    p.t += 1
    if p.v===nothing; p.v = zero(x0); p.m = zero(x0); end
    p.m = p.beta1*p.m .+ (1-p.beta1)*g
    p.v = p.beta2*p.v .+ (1-p.beta2)* g .* g
    m_corrected = p.m/(1 - p.beta1^p.t)
    v_corrected = p.v/(1 - p.beta2^p.t)

    x0 .-= p.lr * (m_corrected ./ (sqrt.(v_corrected) .+ p.epsilon))
    return Jx
end

#AdamX (extension of AMSGrad)
mutable struct AdamX <: Optimiser
    lr::AbstractFloat
    epsilon::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    t::Int
    v
    vhat
    m
end
AdamX(; lr=0.01, epsilon=sqrt(eps()), beta1=0.9, beta2=0.999, t=0, v=nothing, m=nothing, vhat=nothing) = AdamX(lr, epsilon, beta1, beta2, t, v, m, vhat)

function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::AdamX; kwargs...)
    g, Jx = finite_differences_gradient(system, x0, data; kwargs...)
    p.t += 1
    if p.v===nothing; p.v = zero(x0); p.m = zero(x0); vhat = zero(x0); end
    p.m = p.beta1*p.m .+ (1.0-p.beta1)*g
    p.v = p.beta2*p.v .+ (1.0-p.beta2)* g .* g
    if p.t == 1
        p.vhat = +p.v
    else
        p.vhat = max.( ( 1.0 - p.beta1*0.001^(p.t) )^2/( 1.0 - p.beta1*0.001^(p.t-1) )^2 * p.v, p.vhat )
    end

    # b_{1,t-1} = b_{1,t} * lambda^t-1, lambda = 0.001
    x0 .-= p.lr * (p.m ./ (sqrt.(p.vhat) .+ p.epsilon))
    return Jx
end

#Yamadam
mutable struct Yamadam <: Optimiser
    lr::AbstractFloat
    epsilon::AbstractFloat
    beta
    v
    m
    s
    h
end
Yamadam(; lr=1.0, epsilon=sqrt(eps()), v=nothing, m=nothing, s=nothing, h=nothing, beta=nothing) = Yamadam(lr, epsilon, v, m, s, h, beta)

function update!(system::MicroMacroSystem, x0::AbstractArray, data::Data, p::Yamadam; kwargs...)
    g, Jx = finite_differences_gradient(system, x0, data; kwargs...)
    if p.v===nothing; p.v = zero(x0); p.m = zero(x0); p.s = zero(x0); p.h = zero(x0); p.beta = zero(eltype(x0)); end
    p.v = p.beta*p.v .+ (1-p.beta) .* (g .- p.m).^2
    p.m = p.beta*p.m .+ (1-p.beta) .* g
    p.s = p.beta*p.s + (1-p.beta) .* p.h.^2
    h_past = deepcopy(p.h)
    p.h = sqrt.(p.s .+ p.epsilon) ./ sqrt.(p.v .+ p.epsilon) .* p.m
    p.beta = sigmoid( (norm(h_past, 1) + p.epsilon)/(norm(p.h, 1) + p.epsilon) ) - p.epsilon

    x0 .-= p.h
    return Jx
end

### HELPERS ###
sigmoid(z) = 1.0 ./ (1.0 .+ exp(-z))
