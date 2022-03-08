########################## BOUND SEARCH SPACE ##############################
function initial_guess_search(
    system::MicroMacroSystem, 
    x̂0::Array, 
    data::Data;
    α_R::Real, 
    β_R::Real, 
    noise_scaling::Real=0, 
    max_steps::Real=4e5, 
    print_=false, 
    kwargs...
    )

    # Rough threshold
    δ_R = α_R + noise_scaling^2 * β_R
    # number of observations
    T = length(data.macrostates)
    # sampling interval
    m = system.sampling_interval
    # data variance
    variance = var(data.macrostates) # variance of the data

    # preallocation
    iter = 0
    cost = Inf
    cost_aux = Inf

    x0 = copy(x̂0)
    xaux = similar(x0)    
    
    cost_dynamics = Float64[]
    trajectory = typeof(x0)[]
    
    while cost >= δ_R
        
        # Simulate for twice the time of the observation window (for efficieny reasons)
        simulation_2T = integrate(system, x0, 2T)

        for t in 1:(T+1)
            
            iter += 1
            
            if iter >= max_steps      
                println("Exeeded $(iter) number of steps to find initial guess with cost function value under $(δ_R).")
                return xaux, iter, cost_dynamics, trajectory
                break                
            end

            simulation_T = simulation_2T[t:T+t-1]
            x0 = simulation_2T.microstates[ m*(t-1) + 1, : ]
            
            push!(trajectory, x0)
            
            cost = J(simulation_T.macrostates, data.macrostates)/variance

            push!(cost_dynamics, cost)

            if cost < cost_aux
                @inbounds for i in eachindex(x0)
                    xaux[i] = x0[i] # saving auxiliary x0 in case of non-convergence.
                    cost_aux = +cost
                end
            end

            if cost < δ_R 
                if print_
                    println("Roughly approached with J(x̂ᵢ) = $(round( cost[end] , sigdigits=2)) after $iter time steps.")
                end
                return xaux, iter, cost_dynamics, trajectory               
                break
            end

        end

        x0 = simulation_2T.microstates[ m*T + 1 , : ]
    end

    nothing
end

function initial_guess_search(
    system::MicroMacroSystem, 
    x̂0::Array, 
    data::Data,
    num_iters::Int64;
    print_=false, 
    kwargs...
    )

    # number of observations
    T = length(data.macrostates)
    # sampling interval
    m = system.sampling_interval
    # data variance
    variance = var(data.macrostates) # variance of the data

    # preallocation
    cost = Inf
    cost_aux = Inf

    x0 = copy(x̂0)
    xaux = similar(x0)    
    
    cost_dynamics = Float64[]
    trajectory = typeof(x0)[]
    
    iter_aux = 0
    for iter in 1:num_iters
        
        # Simulate for twice the time of the observation window (for efficieny reasons)
        simulation_2T = integrate(system, x0, 2T)

        for t in 1:(T+1)
            
            iter_aux += 1
            
            if iter_aux == num_iters  
                if print_
                    println("Roughly approached with J(x̂ᵢ) = $(round( cost_aux[end] , sigdigits=2)) after $iter_aux time steps.")
                end  
                return xaux, iter_aux, cost_dynamics, trajectory
                break                
            end

            simulation_T = simulation_2T[t:T+t-1]
            x0 = simulation_2T.microstates[ m*(t-1) + 1, : ]
            
            push!(trajectory, x0)
            
            cost = J(simulation_T.macrostates, data.macrostates)/variance

            push!(cost_dynamics, cost)

            if cost < cost_aux
                @inbounds for i in eachindex(x0)
                    xaux[i] = x0[i] # saving auxiliary x0 in case of non-convergence.
                    cost_aux = +cost
                end
            end

        end

        x0 = simulation_2T.microstates[ m*T + 1 , : ]
    end

    nothing
end