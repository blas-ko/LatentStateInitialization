################################## INTEGRATION ################################################

### COMMON TIME STEPPERS ###
function euler!(f!, u, du, params, t, dt)

    f!(du, u, params, t)
    @inbounds for i in eachindex(u)
        u[i] += dt * du[i] # Euler; Should I do rungekutta?
    end
    nothing
end

function rungekutta4!(f!, u, du, params, t, dt)

    k1 = similar(du)
    k2 = similar(du)
    k3 = similar(du)
    k4 = similar(du)

    f!(k1, u, params, t)
    f!(k2, u + k1/2, params, t+dt/2)
    f!(k3, u + k2/2, params, t+dt/2)
    f!(k4, u + k3, params, t+dt)

    @inbounds for i in eachindex(u)
        @inbounds u[i] += dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
    end
    nothing
end

function mapping!(f!, u, du, params, t, dt)
    f!(du, u, params, t)
    @inbounds for i in eachindex(u)
        u[i] = +du[i] # simple map.
    end
    nothing
end

function integrate(system::MicroMacroSystem, x0_array::Vector{Vector{P}}, T::Int64) where P <: Real
    return [ integrate(system, x0, T) for x0 in x0_array ]
end

function integrate(system::MicroMacroSystem, x0::Vector{P}, T::Int64) where P <: Real

    m = system.sampling_interval

    # preallocation
    dx = similar(x0)
    x = deepcopy(x0)
    microstates = Array{P}(undef, m*(T-1)+1, length(x)) # microstates
    macrostates = Array{Float64}(undef, T) # macrostates

    trange = range(0, stop=system.dt*m*(T-1), length=m*(T-1)+1)
    @inbounds microstates[1,:] = x0
    @inbounds macrostates[1] = system.observation_operator(x0)

    for (i, t) in enumerate(trange[2:end])

        system.timestep!( system.model!, x, dx, system.params, t, system.dt )
        @inbounds microstates[i+1,:] = x

        if i % m == 0
            @inbounds macrostates[div(i,m)+1] = system.observation_operator(x)
        end
    end

    # ixrange = range(0, stop=T-1, length=m*(T-1)+1)
    ixrange = range(0, stop=T-1, length=T)
    return Simulation(ixrange, microstates, macrostates)
end