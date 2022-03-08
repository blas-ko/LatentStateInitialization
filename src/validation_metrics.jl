# Maximum prediction time for a time series of T datapoints (such that errors(T) = 2)
function prediction_horizon(errors::AbstractArray{T}, 𝓓::Real=2.0) where T <: Number

    for (i,e) in enumerate(errors)
        if e >= 𝓓
            return i
            break
        end
    end

    println("NRMSE doesn't saturate in this forecasting period. Returning the last index of the forecast...")
    return length(errors)
end

function prediction_horizon(errors_ensemble::AbstractArray{A}, 𝓓::Real=2.0) where A <: AbstractArray
    k_max = Vector{Int64}(undef, length(errors_ensemble))

    for (i,errors) in enumerate(errors_ensemble)
        k_max[i] = prediction_horizon(errors, 𝓓)
    end

    return k_max
end

function prediction_horizon(series1::Data, series2::Data, 𝓓::Real=2.0)
    errors = distance(series1, series2)

    return prediction_horizon(errors, 𝓓)
end

# Computes first out-of-sample error
nowcast_error( errors::Vector, T::Int64 ) = getindex( errors, T+1 )
nowcast_error( errors::Vector, T::Int64, k::Int64 ) = mean([nowcast_error(errors, T + i) for i in 1:k])
nowcast_error( v::Dynamics, u::Dynamics, T::Int64 ) = nowcast_error( distance( v, u ), T )
nowcast_error( v::Dynamics, u::Dynamics, T::Int64, k::Int64 ) = nowcast_error( distance( v,u ), T, k )

# nmse
function distance(v::Simulation, u::Simulation)
    
    @assert size(v.microstates) == size(u.microstates)
    
    timesteps, _ = size(v.microstates)
    
    dist = zeros(timesteps)
    
    for t in 1:timesteps
        dist[t] = msd(v.microstates[t,:], u.microstates[t,:])
        # dist[t] = rmsd(v.microstates[t,:], u.microstates[t,:])
    end
    
    return dist/var(v.microstates)
    # return dist/std(v.microstates)
end

function distance(v::Data, u::Data)
    
    @assert size(v.macrostates) == size(u.macrostates)
    
    return (v.macrostates - u.macrostates).^2/var(v.macrostates)
    # return abs.(v.macrostates - u.macrostates)/std(v.macrostates)
end

function distance(v::AbstractArray, u::AbstractArray)
    
    @assert size(v) == size(u)
    return msd(v,u)/var(v)
    # return rmsd(v,u)/std(v)
end

distance(v::Real, u::Real) = (v - u)^2