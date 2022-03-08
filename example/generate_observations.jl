### GENERATE SYNTHETIC, NOISY, AGGREGATE OBSERVATIONS FRO THE LORENZ SYSTEM 
using Random, Statistics, Distributions
# include("../src/Initialization.jl")
# import .Initialization: integrate

# state_dim = 3
# random_state = rand(state_dim) 
# state_index = rand(200:3200)

# ground_truth_state = Initialization.integrate( system, random_microstate, state_index ).microstates[end,:]

# # define number of observations
# T = 100

# GENERATE SYNTHETIC DATA
# The noise level is latent in principle. We assume we know the noise level.
# noise_level = initialisation_params[:noise_scaling]
# groundtruth, noise, std_data = generate_synthetic_data(system, x0, T, noise_level; seed=seed, return_microstates=true)
# observations = groundtruth.macrostates + noise 

function generate_observations(
    system::Initialization.MicroMacroSystem, 
    x0::AbstractArray, 
    num_observations::Int64, 
    noise_scaling::Real;
    seed=nothing, 
    )
    
    groundtruth = Initialization.integrate(system, x0, num_observations)
    observations = groundtruth.macrostates

    σy = std(observations)
    σN = noise_scaling * σy 
    Gaussian_dist = Normal(0, σN)

    rng = MersenneTwister(seed)
    noise = rand(rng, Gaussian_dist, num_observations)
    # observations_noisy = observations + noise

    observations += noise

    return groundtruth, observations
end

function generate_observations(system, num_observations, noise_scaling; seed=nothing) 
    
    # generate random ground truth microstate
    rng = MersenneTwister(seed)
    random_state = rand(rng, state_dim) 
    random_state_index = rand(rng, 200:3200)
    x0 = Initialization.integrate( system, random_state, random_state_index ).microstates[end,:]

    return generate_observations(system, x0, num_observations, noise_scaling; seed=seed)     
end