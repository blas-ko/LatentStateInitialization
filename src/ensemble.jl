## WARNING: THIS SCRIPT CONTAINS THE FUNCTIONS RELEVANT TO THE ENSEMBLE ANALYSIS SHOWN IN THE PAPER. 
## THIS SCRIPT IS NOT INCLUDED IN THE Initialization.jl MODULE

using DataFrames

# Initialised the `system` for `num_sims` samples in `attractor`    
function run_initialisation_ensemble( 
    system::MicroMacroSystem, 
    attractor::Simulation, 
    T::Int64;
    num_sims=1, 
    seed=nothing, 
    print_=false, 
    save_sequences=false,
    method::Type{U}=Adam, 
    batch_frac=1, learning_rate=0.001, max_iters=10_000,
    α_r, α_R, β_r, β_R, noise_scaling, q=0, 
    kwargs...
    ) where {U <: Optimiser}

    # microstates dimension
    N = size(attractor.microstates)[2]
    
    seed_gt = nothing
    seed_mip = nothing
    
    ## preallocation
    states_type = Vector{eltype(attractor.microstates)}
    observations_type = Vector{eltype(attractor.macrostates)}
    # initialisation stuff
    initial_guess_states = Vector{ states_type }(undef, num_sims)
    rough_states_assim = Vector{ states_type }(undef, num_sims)
    initialised_states_assim = Vector{ states_type }(undef, num_sims)
    rough_states = Vector{ states_type }(undef, num_sims)
    initialised_states = Vector{ states_type }(undef, num_sims)
    fitted_observations_array = Vector{ observations_type }(undef, num_sims)
    init_seeds = Vector(undef, num_sims)
    # data stuff
    data_seeds = Vector(undef, num_sims)
    ground_truth_states_assim = Vector{ states_type }(undef, num_sims)
    ground_truth_states = Vector{ states_type }(undef, num_sims)
    observations_array = Vector{ observations_type }(undef, num_sims)
    noise_array = Vector{ observations_type }(undef, num_sims)
    rough_costs_array = Vector{Vector{Float64}}(undef, num_sims)
    refinement_costs_array = Vector{Vector{Float64}}(undef, num_sims)
    # analysis stuff
    if save_sequences
        rough_states_sequences = Vector{ Vector{ states_type } }(undef, num_sims)
        refinement_states_sequences = Vector{ Vector{ states_type } }(undef, num_sims)
    end

    for sim in 1:num_sims

        if print_
            println("Simulation $(sim):")
        end

        ##### GROUND TRUTH #####

        # SAMPLE GROUND TRUTH MICROSTATES
        if seed != nothing
            seed_gt = seed + sim + 2*num_sims
            x0_assim = rand(MersenneTwister(seed_gt), attractor)
        else
            x0_assim = rand(attractor)
        end
        
        # GENERATE SYNTHETIC DATA
        groundtruth, noise, std_data = generate_synthetic_data(system, x0_assim, T, noise_scaling; seed=seed, return_microstates=true)
        observations = Data(groundtruth) + noise 

        x0 = compute_present_state(system, x0_assim, T)

        ##### INITIALISATION #####

        ## SAMPLE INITIAL GUESS!
        if seed != nothing
            seed_mip = seed + sim
        end
        initial_guess = sample_initial_guess( system, observations; seed=seed_mip)

        ## PREPROCESS OBSERVATIONS!
        observations = moving_average(observations, q)

        ### INITIALISE SYSTEM ###
        results_dict = initialise(
            system, 
            initial_guess, 
            observations,
            α_R=α_R, α_r=α_r, β_R=β_R, β_r=β_r, 
            noise_scaling=noise_scaling, q=q,
            method=method, learning_rate=learning_rate,
            max_iters=max_iters, batch_frac=batch_frac,
            save_sequences=save_sequences,
            print_=print_
            )

        ##### ALLOCATION ##### 

        # initialisation stuff
        initial_guess_states[sim] = results_dict["initial_guess"]
        rough_states_assim[sim] = results_dict["rough_state_assim"]
        initialised_states_assim[sim] = results_dict["initialised_state_assim"]
        rough_states[sim] = results_dict["rough_state"]
        initialised_states[sim] = results_dict["initialised_state"]
        init_seeds[sim] = seed_mip
        # data stuff
        data_seeds[sim] = seed_gt
        ground_truth_states_assim[sim] = x0_assim
        ground_truth_states[sim] = x0
        fitted_observations_array[sim] = results_dict["fitted_observations"]
        observations_array[sim] = groundtruth.macrostates #without processing (add results_dict[4] for preprocessed data)
        noise_array[sim] = noise
        rough_costs_array[sim] = results_dict["rough_costs"]
        refinement_costs_array[sim] = results_dict["refinement_costs"]
        # analysis stuff
        if save_sequences
            rough_states_sequences[sim] = results_dict["rough_sequence"]
            refinement_states_sequences[sim] = results_dict["refinement_sequence"]
        end
    end

    # Dictionary of parameters    
    optimisers_params = Dict(
        :batch_frac => batch_frac,
        :learning_rate => learning_rate,
        :max_iters => max_iters,
        :method => string(method))

    initialisation_params = Dict(
        :α_R => α_R, 
        :α_r => α_r, 
        :β_R => β_R, 
        :β_r => β_r,
        :noise_scaling => noise_scaling,
        :q => q)


    results_summary = Dict(
        "initial_guess_states" => initial_guess_states,
        "rough_states_assim" => rough_states_assim,
        "initialised_states_assim" => initialised_states_assim,
        "rough_states" => rough_states,
        "initialised_states" => initialised_states,
        "fitted_observations_array" => fitted_observations_array,
        "init_seed" => init_seeds,
        "data_seeds" => data_seeds,
        "ground_truth_states_assim" => ground_truth_states_assim,
        "ground_truth_states" => ground_truth_states,
        "observations_array" => observations_array,
        "rough_costs_array" => rough_costs_array,
        "refinement_costs_array" => refinement_costs_array,
        "initialisation_params" => initialisation_params,
        "optimisers_params" => optimisers_params
    )

    if save_sequences
        analysis_sequences = Dict(
            "rough_states_sequences" => rough_states_sequences,
            "refinement_states_sequences" => refinement_states_sequences,
        )
        merge!(results_summary, analysis_sequences)
    end

    if print_
        println("It takes, on average, $( mean( length.(rough_costs_array) ) ) iterations to get under d_R with T = $T .")
        println("It takes, on average, $( mean( length.(refinement_costs_array) ) ) iterations to get under d_r with T = $T .")
    end

    return results_summary

end

function run_initialisation_ensemble( 
    system::MicroMacroSystem, 
    attractor::Simulation, 
    T::Int64,
    num_iters_refine::Int64;
    num_sims=1, 
    seed=nothing, 
    print_=false, 
    save_sequences=false,
    method::Type{U}=Adam, 
    batch_frac=1, learning_rate=0.001, max_iters=10_000,
    α_r, α_R, β_r, β_R, noise_scaling, q=0, 
    kwargs...
    ) where {U <: Optimiser}

    # microstates dimension
    N = size(attractor.microstates)[2]
    
    seed_gt = nothing
    seed_mip = nothing
    
    ## preallocation
    states_type = Vector{eltype(attractor.microstates)}
    observations_type = Vector{eltype(attractor.macrostates)}
    # initialisation stuff
    initial_guess_states = Vector{ states_type }(undef, num_sims)
    rough_states_assim = Vector{ states_type }(undef, num_sims)
    initialised_states_assim = Vector{ states_type }(undef, num_sims)
    rough_states = Vector{ states_type }(undef, num_sims)
    initialised_states = Vector{ states_type }(undef, num_sims)
    fitted_observations_array = Vector{ observations_type }(undef, num_sims)
    init_seeds = Vector(undef, num_sims)
    # data stuff
    data_seeds = Vector(undef, num_sims)
    ground_truth_states = Vector{ states_type }(undef, num_sims)
    ground_truth_states_assim = Vector{ states_type }(undef, num_sims)
    observations_array = Vector{ observations_type }(undef, num_sims)
    noise_array = Vector{ observations_type }(undef, num_sims)
    rough_costs_array = Vector{Vector{Float64}}(undef, num_sims)
    refinement_costs_array = Vector{Vector{Float64}}(undef, num_sims)
    # analysis stuff
    if save_sequences
        rough_states_sequences = Vector{ Vector{ states_type } }(undef, num_sims)
        refinement_states_sequences = Vector{ Vector{ states_type } }(undef, num_sims)
    end

    for sim in 1:num_sims

        if print_
            println("Simulation $(sim):")
        end

        ##### GROUND TRUTH #####

        # SAMPLE GROUND TRUTH MICROSTATES
        if seed != nothing
            seed_gt = seed + sim + 2*num_sims
            x0_assim = rand(MersenneTwister(seed_gt), attractor)
        else
            x0_assim = rand(attractor)
        end
        
        # GENERATE SYNTHETIC DATA
        groundtruth, noise, std_data = generate_synthetic_data(system, x0_assim, T, noise_scaling; seed=seed, return_microstates=true)
        observations = Data(groundtruth) + noise 

        x0 = compute_present_state(system, x0_assim, T)

        ##### INITIALISATION #####

        ## SAMPLE INITIAL GUESS!
        if seed != nothing
            seed_mip = seed + sim
        end
        initial_guess = sample_initial_guess( system, observations; seed=seed_mip)

        ## PREPROCESS OBSERVATIONS!
        observations = moving_average(observations, q)

        ### INITIALISE SYSTEM ###
        results_dict = initialise(
            system, 
            initial_guess, 
            observations,
            num_iters_refine;
            α_R=α_R, β_R=β_R, 
            noise_scaling=noise_scaling, q=q,
            method=method, learning_rate=learning_rate,
            max_iters=max_iters, batch_frac=batch_frac,
            save_sequences=save_sequences,
            print_=print_
            )

        ##### ALLOCATION ##### 

        # initialisation stuff
        initial_guess_states[sim] = results_dict["initial_guess"]
        rough_states_assim[sim] = results_dict["rough_state_assim"]
        initialised_states_assim[sim] = results_dict["initialised_state_assim"]
        rough_states[sim] = results_dict["rough_state"]
        initialised_states[sim] = results_dict["initialised_state"]
        init_seeds[sim] = seed_mip
        # data stuff
        data_seeds[sim] = seed_gt
        ground_truth_states_assim[sim] = x0_assim
        ground_truth_states[sim] = x0
        fitted_observations_array[sim] = results_dict["fitted_observations"]
        observations_array[sim] = groundtruth.macrostates #without processing (add results_dict[4] for preprocessed data)
        noise_array[sim] = noise
        rough_costs_array[sim] = results_dict["rough_costs"]
        refinement_costs_array[sim] = results_dict["refinement_costs"]
        # analysis stuff
        if save_sequences
            rough_states_sequences[sim] = results_dict["rough_sequence"]
            refinement_states_sequences[sim] = results_dict["refinement_sequence"]
        end
    end

    # Dictionary of parameters    
    optimisers_params = Dict(
        :batch_frac => batch_frac,
        :learning_rate => learning_rate,
        :max_iters => max_iters,
        :method => string(method))

    initialisation_params = Dict(
        :α_R => α_R,  
        :β_R => β_R, 
        :num_iters_refine => num_iters_refine,
        :noise_scaling => noise_scaling,
        :q => q)

    results_summary = Dict(
        "initial_guess_states" => initial_guess_states,
        "rough_states_assim" => rough_states_assim,
        "initialised_states_assim" => initialised_states_assim,
        "rough_states" => rough_states,
        "initialised_states" => initialised_states,
        "fitted_observations_array" => fitted_observations_array,
        "init_seed" => init_seeds,
        "data_seeds" => data_seeds,
        "ground_truth_states_assim" => ground_truth_states_assim,
        "ground_truth_states" => ground_truth_states,
        "observations_array" => observations_array,
        "rough_costs_array" => rough_costs_array,
        "refinement_costs_array" => refinement_costs_array,
        "initialisation_params" => initialisation_params,
        "optimisers_params" => optimisers_params
    )

    if save_sequences
        analysis_sequences = Dict(
            "rough_states_sequences" => rough_states_sequences,
            "refinement_states_sequences" => refinement_states_sequences,
        )
        merge!(results_summary, analysis_sequences)
    end

    if print_
        println("It takes, on average, $( mean( length.(rough_costs_array) ) ) iterations to get under d_R with T = $T .")
        println("After $num_iters_refine epochs, the cost is  $(refinement_costs_array[end]) using $method with T = $T .")
    end

    return results_summary

end

function analyse_ensemble(system::MicroMacroSystem, results::Dict, prediction_steps::Int64=1000; save_simulations=false)

    # number of observations
    T = length(results["observations_array"][1])
    m = system.sampling_interval

    sims_gt    = integrate(system, results["ground_truth_states_assim"], T+prediction_steps)
    sims_star  = integrate(system, results["initialised_states_assim"], T+prediction_steps)
    sims_rough = integrate(system, results["rough_states_assim"], T+prediction_steps);

    errors_star_obs_space    = distance.( Data.(sims_gt), Data.(sims_star) )
    errors_rough_obs_space   = distance.( Data.(sims_gt), Data.(sims_rough) )
    errors_star_model_space  = distance.( sims_gt, sims_star )
    errors_rough_model_space = distance.( sims_gt, sims_rough );

    rough_lengths = length.( results["rough_costs_array"] )
    refinement_lengths = length.( results["refinement_costs_array"] )

    norm_x_xstar_assim = distance.( results["ground_truth_states_assim"], results["initialised_states_assim"] )
    norm_x_xR_assim = distance.( results["ground_truth_states_assim"], results["rough_states_assim"] )

    norm_x_xstar = distance.( results["ground_truth_states"], results["initialised_states"] )
    norm_x_xR = distance.( results["ground_truth_states"], results["rough_states"] )

    norm_y_yR = distance.( results["observations_array"], [y.macrostates[1:T] for y in sims_rough] )
    norm_y_ystar = distance.( results["observations_array"], results["fitted_observations_array"])

    # prediction_horizon_obs_space   = prediction_horizon( mean(errors_star_obs_space)[T:end] ) #kmax option a)
    # prediction_horizon_model_space = Int( ceil( prediction_horizon( mean(errors_star_model_space)[m*T:end] ) ) )

    prediction_horizons_obs_space   = prediction_horizon.( [ e[T:end] for e in errors_star_obs_space] )
    prediction_horizons_model_space = prediction_horizon.( [e[m*T:end] for e in errors_star_model_space] ) 

    miters = results["optimisers_params"][:max_iters]
    αr = results["initialisation_params"][:α_r]
    βr = results["initialisation_params"][:β_r]
    noise_scaling = results["initialisation_params"][:noise_scaling] 
    suc_rate = success_rate( 
        results["refinement_costs_array"]; 
        max_iters=miters, α_r=αr, β_r=βr, noise_scaling=noise_scaling)
    
    nowcast_errors_model_space = nowcast_error.( errors_star_model_space, m*T, m)
    nowcast_errors_obs_space = nowcast_error.( errors_star_obs_space, T)

    delta_R, delta_r = compute_threshold_parameters(; results["initialisation_params"]... )

    results_summary = Dict(
        "num_observations" => T,
        "errors_star_obs_space" => errors_star_obs_space,
        "errors_star_model_space" => errors_star_model_space,
        "errors_rough_obs_space" => errors_rough_obs_space,
        "errors_rough_model_space" => errors_rough_model_space,
        "rough_lengths" => rough_lengths,
        "refinement_lengths" => refinement_lengths,
        "norm_x_xstar_assim" => norm_x_xstar_assim,
        "norm_x_xR_assim" => norm_x_xR_assim,
        "norm_x_xstar" => norm_x_xstar,
        "norm_x_xR" => norm_x_xR,
        "norm_y_yR" => norm_y_yR,
        "norm_y_ystar" => norm_y_ystar,
        "prediction_horizons_obs_space" => prediction_horizons_obs_space,
        "prediction_horizons_model_space" => prediction_horizons_model_space,
        "success_rate" => suc_rate,
        "nowcast_errors_obs_space" => nowcast_errors_obs_space,
        "nowcast_errors_model_space" => nowcast_errors_model_space,
        "delta_R" => delta_R,
        "delta_r" => delta_r,            
    )

    if save_simulations
        analysis_results = Dict(
            "sims_gt" => sims_gt,
            "sims_rough" => sims_rough,
            "sims_star" => sims_star
        )
        merge!(results_summary, analysis_results)
    end

    return results_summary
end

function varying_num_observations(num_obs_range::AbstractArray{U}, args...; kwargs...) where {U <: Int}

    system = args[1]

    ## preallocation
    df_varying_T = DataFrame(
        T = Int[],
        N = Int[],
        success_rate = Float64[],
        # prediction_horizon = Int[],
        prediction_horizon = Float64[],
        prediction_horizon_std = Float64[],
        nowcast_error = Float64[],
        nowcast_error_std = Float64[],
        # nowcast_error_model = Float64[],
        # nowcast_error_model_std = Float64[],
        norm_x_xstar = Float64[],
        norm_x_xstar_std = Float64[],
        rough_length = Float64[],
        rough_length_std = Float64[],
        refinement_length = Float64[],
        refinement_length_std = Float64[],
        delta_r = Float64[] )

    for T in num_obs_range 

        println("Initialising ensemble for time series of $T observations.")

        # initialise ensemble
        results_dict = run_initialisation_ensemble(args[1:2]..., T; kwargs..., 
            print_=false, save_sequences=false)

        # process results
        pred_steps = 2500
        analysis_dict = analyse_ensemble(system, results_dict, pred_steps)

        pred_hors = [ prediction_horizon(error[T:end]) for error in analysis_dict["errors_star_obs_space"] ]

        # save relevant quantities
        sr   = analysis_dict["success_rate"]
        # ph   = analysis_dict["prediction_horizon_obs_space"] # kmax option a)
        ph   = mean( analysis_dict["prediction_horizons_obs_space"] ) #kmax definition
        phs  = std( analysis_dict["prediction_horizons_obs_space"] )
        ne   = median( analysis_dict["nowcast_errors_obs_space"] )
        nes  = std( analysis_dict["nowcast_errors_obs_space"] )
        nxs  = median( analysis_dict["norm_x_xstar"] )
        nxss = std( analysis_dict["norm_x_xstar"] )
        rol  = median( analysis_dict["rough_lengths"] )
        rols = std( analysis_dict["rough_lengths"] )
        rel  = median( analysis_dict["refinement_lengths"] )
        rels = std( analysis_dict["refinement_lengths"] )
        delr = analysis_dict["delta_r"]
        N = length(results_dict["ground_truth_states"][1])

        push!(df_varying_T,  (T, N, sr, ph, phs, ne, nes, nxs, nxss, rol, rols, rel, rels, delr) )

    end

    return df_varying_T
end

function varying_num_obs_and_sampling_interval(num_obs_range::AbstractArray{U}, sampling_interval_range::AbstractArray{U},
    args...; kwargs...) where {U <: Int}

    system = args[1]
    original_m = system.sampling_interval
    println("Original sampling interval: $original_m")

    m_vs_T_vs_E0        = Matrix{Float64}(undef, length(sampling_interval_range), length(num_obs_range))
    m_vs_T_vs_Exstar    = Matrix{Float64}(undef, length(sampling_interval_range), length(num_obs_range))
    # m_vs_T_vs_kmax      = Matrix{Float64}(undef, length(sampling_interval_range), length(num_obs_range))
    m_vs_T_vs_kmax      = Matrix{Float64}(undef, length(sampling_interval_range), length(num_obs_range))
    m_vs_T_vs_success   = Matrix{Float64}(undef, length(sampling_interval_range), length(num_obs_range))
    N = Inf

    for (i,m) in enumerate(sampling_interval_range)

        system.sampling_interval = m
        println("Considering system for a sampling interval of $m .")

        df = varying_num_observations(num_obs_range::AbstractArray{U}, system, args[2:end]...; kwargs...)

        m_vs_T_vs_E0[i, :]        = df[!, "nowcast_error"]
        m_vs_T_vs_Exstar[i, :]    = df[!, "norm_x_xstar"]
        m_vs_T_vs_kmax[i, :]      = df[!, "prediction_horizon"]
        # m_vs_T_vs_kmax_mean[i, :] = df[!, "prediction_horizon_median"]
        m_vs_T_vs_success[i, :]   = df[!, "success_rate"]
        N = df.N[1]

    end

    results_dict = Dict(
        "num_observations_array" => num_obs_range,
        "microstates_dimension" => N,
        "sampling_interval_array" => sampling_interval_range,
        "nowcast_error_matrix" => m_vs_T_vs_E0,
        "microstates_error_matrix" => m_vs_T_vs_Exstar,
        "prediction_horizon_matrix" => m_vs_T_vs_kmax,
        # "prediction_horizon_mean_matrix" => m_vs_T_vs_kmax_mean,
        "success_rate_matrix" => m_vs_T_vs_success
    )

    system.sampling_interval = original_m
    return results_dict
end

function varying_rough_threshold(alphas::AbstractArray, betas::AbstractArray, args...; kwargs...)

    @assert length(alphas) == length(betas)

    system = args[1]

    ## preallocation
    dict_varying_threshold = Dict()

    norm_x_xstar_array = Vector{Float64}[]
    norm_x_xR_array = Vector{Float64}[]

    norm_x_xstar_assim_array = Vector{Float64}[]
    norm_x_xR_assim_array = Vector{Float64}[]

    norm_y_yR_array = Vector{Float64}[]
    norm_y_ystar_array = Vector{Float64}[]

    rough_lengths_array = Vector{Int64}[]
    refinement_lengths_array = Vector{Int64}[]

    success_rate_array = Float64[]
    delta_R_array = Float64[]
    delta_r_array = Float64[]


    for i in eachindex(alphas) 
        println("$i-th alpha = $(alphas[i])")
        results_dict = run_initialisation_ensemble(args...; kwargs..., 
            print_=false, save_sequences=false, α_R=alphas[i], β_R=betas[i])

        # process results
        pred_steps = 10
        analysis_dict = analyse_ensemble(system, results_dict, pred_steps)

        push!(norm_x_xR_assim_array, analysis_dict["norm_x_xR_assim"])
        push!(norm_x_xstar_assim_array, analysis_dict["norm_x_xstar_assim"])
        push!(norm_x_xR_array, analysis_dict["norm_x_xR"])
        push!(norm_x_xstar_array, analysis_dict["norm_x_xstar"])
        push!(norm_y_yR_array, analysis_dict["norm_y_yR"])
        push!(norm_y_ystar_array, analysis_dict["norm_y_ystar"])
        push!(rough_lengths_array, analysis_dict["rough_lengths"])
        push!(refinement_lengths_array, analysis_dict["refinement_lengths"])
        push!(success_rate_array, analysis_dict["success_rate"])
        push!(delta_R_array, analysis_dict["delta_R"])
        push!(delta_r_array, analysis_dict["delta_r"])

    end

    dict_varying_threshold = Dict(
        "norms_x_xR_assim" => norm_x_xR_assim_array,
        "norms_x_xstar_assim" => norm_x_xstar_assim_array,
        "norms_x_xR" => norm_x_xR_array,
        "norms_x_xstar" => norm_x_xstar_array,
        "norms_y_yR" => norm_y_yR_array,
        "norms_y_ystar" => norm_y_ystar_array,
        "rough_lengths" => rough_lengths_array,
        "refinement_lengths" => refinement_lengths_array, 
        "success_rates" => success_rate_array,
        "delta_R" => delta_R_array,
        "delta_r" => delta_r_array )

    return dict_varying_threshold
end

function varying_optimiser(optimisers::Vector, num_iters_refine::Int64, args...; kwargs...) #where {U <: Optimiser}
    ## preallocation

    system = args[1]

    df_cost_trajectories = DataFrame( )

    df_varying_optimiser = DataFrame(
        optimiser = String[],
        refinement_cost = Float64[],
        refinement_cost_std = Float64[],
        microstates_error = Float64[],
        microstates_error_std = Float64[],
        microstates_assim_error = Float64[],
        microstates_assim_error_std = Float64[],
        refinement_length = Float64[],
        refinement_length_std = Float64[],
        success_rate = Float64[]    
        )

    for optimiser in optimisers 

        println("Running ensemble for $optimiser optimiser.")

        results_dict = run_initialisation_ensemble(args...; kwargs..., 
            print_=false, save_sequences=false, method=optimiser)

        results_dict_fixed_iters = run_initialisation_ensemble(args..., num_iters_refine; kwargs..., 
            print_=false, save_sequences=false, method=optimiser)

        # process results
        pred_steps = 100
        analysis_dict = analyse_ensemble(system, results_dict, pred_steps)

        # Vector quantities
        average_cost_sequence     = median(results_dict_fixed_iters["refinement_costs_array"])
        std_cost_sequence         = std(results_dict_fixed_iters["refinement_costs_array"])

        # numerical quantities
        average_refinement_cost   = median([costs[end] for costs in results_dict["refinement_costs_array"]])
        std_refinement_cost       = std([costs[end] for costs in results_dict["refinement_costs_array"]])
        average_microstates_error = median(analysis_dict["norm_x_xstar"])
        std_microstates_error     = std(analysis_dict["norm_x_xstar"])
        average_microstates_assim_error = median(analysis_dict["norm_x_xstar_assim"])
        std_microstates_assim_error     = std(analysis_dict["norm_x_xstar_assim"])
        average_refinement_length = median(analysis_dict["refinement_lengths"])
        std_refinement_length     = std(analysis_dict["refinement_lengths"])
        suc_rate                  = analysis_dict["success_rate"]

        df_cost_trajectories[!,string(optimiser)] = average_cost_sequence
        df_cost_trajectories[!,string(optimiser)*"_std"] = std_cost_sequence 

        push!(df_varying_optimiser,  
               (string(optimiser), 
                average_refinement_cost, 
                std_refinement_cost, 
                average_microstates_error,
                std_microstates_error,
                average_microstates_assim_error,
                std_microstates_assim_error,
                average_refinement_length,
                std_refinement_length,
                suc_rate) 
            )

    end

    return df_varying_optimiser, df_cost_trajectories
end

## HELPER FUNCTIONS ## 

# Compute fraction of sucessful initialisations
function success_rate(
    refinement_costs_array::Vector{Vector{U}};
    max_iters::Int64, 
    α_r::Real, β_r::Real, noise_scaling::Real) where {U <: Real}

    # 1. - A = | refinements of length max_iters |
    # 2. - B = | cost > 2*delta_r |
    # return intersection(A,B)

    δ_r = α_r + noise_scaling^2 * β_r

    lengths = length.(refinement_costs_array)
    final_costs = [ v[end] for v in refinement_costs_array  ]
    unsuccessful_costs = findall(>(2*δ_r), final_costs)
    unsuccessful_lengths = findall(>(max_iters), lengths)

    # println("lengths: " , sum(unsuccessful_costs), " ", sum(unsuccessful_lengths))

    unsuccessful_inits = length(intersect( unsuccessful_costs, unsuccessful_lengths ))

    return 1 .- (unsuccessful_inits ./ length( refinement_costs_array  ))
end

flatten( v::Vector{Vector{U}} ) where {U <: Any} = collect( Iterators.flatten(v) )