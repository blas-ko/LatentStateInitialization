# include("./Initialisation.jl")
using DynamicalSystems, FFTW
import ChaosTools: kaplanyorke_dim
import DynamicalSystems: lyapunovspectrum
import Statistics: median

# Returns observations with all its useful quantities
function generate_synthetic_data(
    system::MicroMacroSystem, 
    x0::AbstractArray, 
    T::Int64, 
    noise_scaling::Real;
    seed=nothing, 
    return_microstates::Bool=false)
    
    groundtruth = integrate(system, x0, T)
    observations = Data(groundtruth)

    σy = std(observations.macrostates)
    σN = noise_scaling * σy 

    Gaussian_dist = Normal(0, σN)

    rng = MersenneTwister(seed)
    noise = rand(rng, Gaussian_dist, T)
    # observations_noisy = observations + noise

    if return_microstates
        return groundtruth, noise, σy
    else
        return observations, noise, σy
    end

end

function generate_synthetic_data(system, T, noise_scaling; seed, return_microstates) 
    return generate_synthetic_data(system, rand(system.state_space_dim), T, noise_scaling; seed=seed, return_microstates=return_microstates)
end

# Sample initial guess at random (works for homogenous observation operators)
function sample_initial_guess(observation_operator::Function, data::Data, state_dimension::Int64; seed=nothing)
    
    y_0 = data.macrostates[1]
    initial_guess = rand(MersenneTwister(seed), state_dimension)
    
    return initial_guess * y_0 / observation_operator( initial_guess )
end

function sample_initial_guess(system::MicroMacroSystem, data::Data; seed=nothing)
    
    return sample_initial_guess(system.observation_operator, data, system.state_space_dim; seed=seed)
end

# From the assimilative microstates, compute the present time microstates
function compute_present_state(system, x_assim, T::Number)
    return integrate(system, x_assim, T).microstates[end,:] # [end+1-system.sampling_interval,:]
end
compute_present_state(system, x_assim, data::Data) = compute_present_state(system, x_assim, length(data.macrostates))


### MEASURES AND METRICS OF THE SYSTEM

# Array of distances between random points in the attractor
function attractor_distances(attractor::Dynamics, n_samples=1000)
    ## Compute the distance between pairs of points in the attractor
    
    distances = Float64[]
    
    for i in 1:n_samples
        
        x_i = rand(attractor.macrostates)
        
        for j in 1:n_samples
            
            x_j = rand(attractor.macrostates)

            push!(distances, distance(x_i, x_j) )

        end
    end
    
    return distances
end

# Inverse of lyapunov's largest exponent in units of number of samples in obs space.
function lyapunov_time(
    system::MicroMacroSystem, 
    # x0::AbstractArray, 
    DS::Type{U}=DiscreteDynamicalSystem;
    tot_time::Number=500_000) where {U <: DynamicalSystem}

    ds = DS( system.model!, rand(system.state_space_dim), system.params )

    max_lyap_exp = lyapunov(ds, tot_time)
        
    # t_λ = log(10) / ( system.sampling_interval * max_lyap_exp )
    t_λ = log(10) / ( system.sampling_interval * max_lyap_exp * system.dt )  
    return t_λ

end

# attractor's dimension. This might be time consuming (precomplie first using num_QR_decompositions=1)
function kaplanyorke_dim(
    system::MicroMacroSystem, 
    # x0::AbstractArray, 
    DS::Type{U}=DiscreteDynamicalSystem,
    num_QR_decompositions::Int64=20_000;
    transient_time::Number=100,
    return_spectrum=true) where {U <: DynamicalSystem}

    lyap_spectrum = lyapunovspectrum( system, DS, num_QR_decompositions; Ttr=transient_time )

    dim_KY = kaplanyorke_dim( sort(lyap_spectrum, rev=true) )

    if return_spectrum
        return dim_KY, lyap_spectrum
    else
        return dim_KY
    end

end

# kolmogorov-sinai entropy

function lyapunovspectrum(
    system::MicroMacroSystem, 
    # x0::AbstractArray, 
    DS::Type{U}=DiscreteDynamicalSystem,
    num_QR_decompositions::Int64=20_000;
    transient_time::Number=100) where {U <: DynamicalSystem}

    ds = DS(system.model!, rand(system.state_space_dim), system.params)
    return lyapunovspectrum(ds, num_QR_decompositions; Ttr=transient_time)
end

## power spectra
function powerspectra(y::Data)

    n_samples = length(y)
    dt = y.time[2] - y.time[1]
    
    y_ft  = fft(y.macrostates)
    freqs = range(0, stop=1/(2dt) , length=n_samples÷2  )
    
    S = (2/n_samples * norm.(y_ft)[1:n_samples÷2]).^2
    return freqs, S
end

function powerspectra(system::MicroMacroSystem, y::Simulation)
    
    n_observations = length(y.macrostates)
    n_samples = size(y.microstates)[1]
    
    y_micro = system.observation_operator( y.microstates )
    times_micro = range(y.time[1], stop=y.time[end], length=n_samples)
    
    y_micro = Data(times_micro, y_micro)
    
    freqs_obs, y_spectra_obs = powerspectra( Data(y) )
    freqs_micro, y_spectra_micro = powerspectra( y_micro )
    
    # normalise frequencies
    # freqs_obs   = freqs_obs ./(system.dt * system.sampling_interval)
    # freqs_micro = freqs_micro ./(system.dt * system.sampling_interval)

    return ( (freqs_obs, y_spectra_obs), (freqs_micro, y_spectra_micro) )
end 

# 2^15 = 32768
function powerspectra(system::MicroMacroSystem, sample_size::Int64; transient_time=250) 

    x0 = integrate( system, rand(system.state_space_dim), transient_time ).microstates[end,:]
    y = integrate( system, x0, sample_size )
    return powerspectra(system, y)
end

# 2^12 = 4096
function powerspectra(system::MicroMacroSystem, sample_size::Int64, num_avgs::Int64; transient_time=250) 

    (f_obs, s_obs), (f_micro, s_micro) = powerspectra(system, sample_size; transient_time = transient_time)

    avg_spectra_obs = deepcopy(s_obs)
    avg_spectra_micro = deepcopy(s_micro)

    for i in 1:num_avgs
        (_, s_obs), (_, s_micro) = powerspectra(system, sample_size; transient_time = transient_time) 

        avg_spectra_obs += s_obs
        avg_spectra_micro += s_micro
    end

    avg_spectra_micro = avg_spectra_micro/num_avgs 
    avg_spectra_obs   = avg_spectra_obs/num_avgs

    return ( (f_obs, avg_spectra_obs), (f_micro, avg_spectra_micro) )
end

### Other helper functions
function sigmoid_parametrised(x; x1=0, x2=log(2), y1=1/2, y2=1/3)
        
    a = log( (inv(y2) - 1)/(inv(y1) - 1) )/(x1-x2)
    b = log(inv(y1) - 1) + a*x1
    
    return inv( 1 + exp(-a*x + b) )
end
sigmoid_errors(x, δr) = sigmoid_parametrised(x; x1=1-δr, y1=δr, x2=δr, y2=1-δr)
sigmoid_success_rate(x, ϵ=1e-2) = sigmoid_parametrised(x; x1=ϵ, y1=ϵ, x2=1-ϵ, y2=1-ϵ)

function compute_threshold_parameters(; α_R::Real, α_r::Real, β_R::Real, β_r::Real, noise_scaling::Real, kwargs...)
    
    δ_R = α_R + noise_scaling^2 * β_R
    δ_r = α_r + noise_scaling^2 * β_r
    
    return δ_R, δ_r
end

function add_filename_metadata()
    #stuff
    return string()
end

function median( u::Vector{Vector{U}} ) where U 

    u_median = hcat(u...)
    u_median = median(u_median, dims=2)
    return reshape( u_median, length(u_median) )
end