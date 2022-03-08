####################### COST FUNCTION MINIMISATION #############################

## g-TIME-WEIGTHED LEAST-SQUARES COST FUNCTION ##
# J(model, data) = dot(model - data, model - data)
# J(model::AbstractArray, data::AbstractArray) = msd(model, data)#/var(data)
J(model::AbstractArray, data::AbstractArray) = mean( abs2.(model - data) )#/var(data)
J(model::Dynamics, data::Dynamics) = J(model.macrostates, data.macrostates)

function finite_differences_gradient(system::MicroMacroSystem, x0::Vector{P}, data::Data; batch_size=nothing, kwargs...) where {P <: Real}

    # From kwargs, we need batch_size only

    ε = convert(P, sqrt(eps())) # gradient variation
    N = length(x0) # microstate dimensionality
    T = length(data.macrostates) # number of measurements
    σ² = var(data.macrostates) # variance of the data

    if batch_size == nothing
        batch_size = N 
    end

    # Simulation of model with no variation
    model_x0 = integrate(system, x0, T)
            # params=params, dt=dt, m=m, problem_type=problem_type)

    # Cost Function of unvaried simulation
    cost = J(model_x0, data)

    # Allocation of variation of cost function
    cost_gradient = zeros( typeof(cost), N )

    # evaluation of cost function for variation dε of a random sample of components of x0.
    @inbounds for i in sample( 1:N, batch_size )

        x0[i]   += ε # vary it's component by ε
        model_dε = integrate(system, x0, T)
        x0[i]   -= ε # Return x0 to its original state

        x0[i]   -= ε
        model_dεm = integrate(system, x0, T)
        x0[i]   += ε # if we wanted to do both sides...

        cost_gradient[i] = J(model_dε, data) - J(model_dεm, data)
    end

    return cost_gradient/(2*ε*σ²), cost/σ²
end

######################### COST FUNCTION MINIMIZATION ###########################

# Here, the procedure repeats until dl is achieved.
function cost_minimization(
    system::MicroMacroSystem, 
    x0::Array{P}, 
    data::Data;
    α_r::Real, 
    β_r::Real, 
    noise_scaling::Real=0, 
    method::Type{U}=Adam, 
    learning_rate=0.01, 
    max_iters=10_000, 
    batch_frac=1, 
    print_::Bool=false, 
    kwargs...
    ) where {P <: Real, U <: Optimiser}

    # kwargs: δ_r, max_iters, batch_size, 

    # δ_r = kwargs[:δ_r]
    # max_iters = kwargs[:max_iters]

    # Compute refinement threshold
    δ_r = α_r + noise_scaling^2 * β_r
    # Data variance
    σ² = var(data.macrostates)
    # Number of observations
    T = length(data)
    # Microstates dimension
    N = length(x0)
    # Number of microstates varied for finite differences
    batch_size = Int( round(N * batch_frac) )
    # Instantiate optimiser
    optimiser = method(lr=learning_rate)

    ## preallocation
    x̂0_vector = Vector{P}[]
    cost_vector = Float64[]

    # initial cost
    cost = J( integrate(system, x0, T).macrostates, data.macrostates )/σ²

    push!(x̂0_vector, x0)
    push!(cost_vector, cost)

    x̂ᵢ = copy(x0)

    cnt = 1
    while cost >= δ_r
        # println(cost)

        cost = update!(system, x̂ᵢ, data, optimiser; batch_size=batch_size)

        # println(cost)
        push!(cost_vector, cost)
        push!(x̂0_vector, +x̂ᵢ)

        if cnt == max_iters
            if print_
                println("Couldn't converge below $δ_r after $max_iters steps. \nThe current cost value is $cost . Breaking...")
            end
            break
        end

        cnt += 1
        if isnan(cost) # Maybe we can use this case better...
            println(cost)
            println("Cost diverged... breaking...")
            return [ zeros(length(x0)) ], [NaN]
            break
        end

    end

    if print_
        println("Converged to J(x̂ᵢ) = $(round( cost[end] , sigdigits=2)) after $cnt steps with $(method) method.")
    end
    return x̂0_vector, cost_vector
end

# Here, the number of descent iterations is fixed by num_iters
function cost_minimization(
    system::MicroMacroSystem, 
    x0::Array{P}, 
    data::Data, 
    num_iters::Int64;
    method::Type{U}=Adam, 
    learning_rate=0.01, 
    batch_frac=1, 
    print_::Bool=false, 
    kwargs...
    ) where {P <: Real, U <: Optimiser }

    # Data variance
    σ² = var(data.macrostates)
    # Number of observations
    T = length(data)
    # Microstates dimension
    N = length(x0)
    # Number of microstates varied for finite differences
    batch_size = Int( round(N * batch_frac) )
    # Instantiate optimiser
    optimiser = method(lr=learning_rate)

    ## preallocation
    x̂0_vector = Vector{Vector{P}}(undef, num_iters)
    cost_vector = Vector{Float64}(undef, num_iters)

    # initial cost
    cost = J( integrate(system, x0, T).macrostates, data.macrostates )/σ²

    x̂0_vector[1] = x0
    cost_vector[1] = cost

    x̂ᵢ = copy(x0)

    for i in 2:(num_iters)

        cost_vector[i] = update!(system, x̂ᵢ, data, optimiser; batch_size=batch_size)
        x̂0_vector[i] = +x̂ᵢ

        # if print_
        #     println("$(i)-th iteration: J(x̂ᵢ) = $(round( cost[i-1] , sigdigits=2))")
        # end

        if isnan(cost_vector[i]) # Maybe we can use this case better...
            # println(cost[i:-1:1])
            println("Cost diverged... breaking...")
            return [ zeros(length(x0)) ], [NaN]
            break
        end

    end

    if print_
        println("Converged to J(x̂ᵢ) = $(round( cost[end] , sigdigits=2)) after $num_iters steps with $(method) method.")
    end
    return x̂0_vector, cost_vector
end
