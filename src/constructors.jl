########## DATA STRUCTURES TO WORK IN THE INITIALISATION SCHEME ################

### SYSTEM TYPE: USEFUL FOR INTEGRATION ###
mutable struct MicroMacroSystem 
    
    model!::Function
    observation_operator::Function
    sampling_interval::Int64
    params::Union{Tuple, Array}
    state_space_dim::Int64
    
    dt::Real
    timestep!::Function
    
end 

function MicroMacroSystem(
    model!, 
    observation_operator, 
    sampling_interval, 
    params)  
        
    return MicroMacroSystem(model!, observation_operator, sampling_interval, params, 1, mapping!)
end

# Dynamics types: Data (rebuilds real data into a type), Simulation (time, micro, macro)
abstract type Dynamics; end

mutable struct Data{U<:Real, Q<:Real} <: Dynamics
    time::AbstractRange{U}
    macrostates::Array{Q}
end

mutable struct Simulation{U<:Real, P<:Any, Q<:Real} <: Dynamics
    time::AbstractRange{U}
    microstates::Array{P}
    macrostates::Array{Q}
end

############################# INTERPOLATION ####################################

# linear interpolation of a macrostate into an specific time t. Always return a Data object even when input is simulation
function linear_interpolation(D::U, τ::Real) where U <: Dynamics
    t₀ = D.time[1]
    @assert (round(τ, digits=1) <= D.time[end] &&  round(τ, digits=1) >= t₀) "τ is not in the time-range of the solution."

    τ = promote(τ, t₀)[1]
    Δt = D.time[2] - t₀

    if τ ∈ D.time
        i = Int( round((τ-t₀)/Δt) )+1
        return D.macrostates[i,:]
    else
        i = Int(floor((τ-t₀)/Δt))+1

        if i >= length(D.macrostates)
            return D.macrostates[i,:] + (D.macrostates[end,:] - D.macrostates[end-1,:])/Δt * (τ - D.time[i])
        else
            return D.macrostates[i,:] + (D.macrostates[i+1,:] - D.macrostates[i,:])/Δt * (τ - D.time[i])
        end

    end

end

# linear interpolation of a macrostate in a time range τrange.
function linear_interpolation(D::U, τrange::AbstractArray) where U <: Dynamics
    macrostates = Array{eltype(D.macrostates)}( undef, length(τrange)) #, size(D.macrostates,2) )

    for (i, τ) in enumerate(τrange)
            macrostates[i,:] = D(τ)
    end

    return Data(τrange, macrostates)
end

# Make interpolation callable as a function
for Type in [Simulation, Data]
    (D::Type)(τ::Real) = linear_interpolation(D, τ)
    (D::Type)(τrange::AbstractArray) = linear_interpolation(D, τrange)
end

###################### OUTER CONSTRUCTORS OF INITIALISATION TYPE ####################

# Convert from Simulation type (which includes microstates) to Data type
# (it changes the time ticks accordingly to the macrostates evolution)
function Data(D::Simulation)
    m = div( length(D.time), length(D.macrostates))
    return Data(D.time[1:m:end], D.macrostates)
end
# Transforms array into Data type with indication time-indexes
Data(D::Array{Q}) where Q <: Real = Data(0:length(D)-1, D)

function getindex(S::Simulation, U::Union{UnitRange{Int64}, Int64})
    Stmp = Data(S)
    return Data(Stmp.time[U] , Stmp.macrostates[U])
end

function getindex(S::Data, U::Union{UnitRange{Int64}, Int64})
    return Data(S.time[U] , S.macrostates[U])
end

lastindex(D::Dynamics) = last( eachindex(D.macrostates) )

# take random microstates from Simulation object
function rand(simu::Simulation)
    i = rand(1:length(simu.macrostates))

    return simu.microstates[i,:]
end

function rand(rng::MersenneTwister, simu::Simulation)
    i = rand(rng, 1:length(simu.macrostates))

    return simu.microstates[i,:]
end

length(D::Data) = length(D.macrostates)

# these always return a Data object
for T in [:Data, :Simulation]
    for S in [:Data, :Simulation]
        for op in (:+, :-)

        @eval begin
                function ($op)(x::$T, y::AbstractArray)
                    @assert size(x.macrostates) == size(y)
                    return Data(x.time, $op(x.macrostates, y))
                end

                function ($op)(x::AbstractArray, y::$T)
                    @assert size(x) == size(y.macrostates)
                    return Data(y.time, $op(x, y.macrostates))
                end

                function dot(x::AbstractArray, y::$T)
                    @assert size(x) == size(y.macrostates)
                    return dot(x, y.macrostates)
                end

                function dot(x::$T, y::AbstractArray)
                    @assert size(x.macrostates) == size(y)
                    return dot(x.macrostates, y)
                end


                function ($op)(x::$T, y::$S)
                    @assert x.time ≈ y.time
                    return Data(x.time, $op(x.macrostates, y.macrostates))
                end

                function dot(x::$T, y::$S)
                    @assert x.time ≈ y.time
                    return dot(x.macrostates, y.macrostates)
                end
            end

        end
    end
end
