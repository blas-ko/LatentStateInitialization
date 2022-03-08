# main function
function initialise(
    system::MicroMacroSystem, 
    x0::Array{P}, 
    observations::Data;
    α_R::Real,
    α_r::Real, 
    β_R::Real,
    β_r::Real, 
    noise_scaling::Real=0, 
    q::Int64=0,
    method::Type{U}=Adam, 
    learning_rate=0.001, 
    max_iters=10_000, 
    batch_frac=1, 
    print_::Bool=false, 
    save_sequences::Bool=false,
    kwargs... 
    ) where {P <: Real, U <: Optimiser}


    ## PREPROCESS OBSERVATIONS!
    observations = moving_average(observations, q)

    ## BOUND SEARCH!
    x_R_assim, rough_num_steps, rough_costs, rough_sequence = initial_guess_search(system, x0, observations; 
            print_=print_, α_R=α_R, β_R=β_R, noise_scaling=noise_scaling)
    
    ## REFINE MICROSTATES!
    refinement_sequence, refinement_costs = cost_minimization(system, x_R_assim, observations;
            print_=print_, method=method, α_r=α_r, β_r=β_r, noise_scaling=noise_scaling,
            learning_rate=learning_rate, max_iters=max_iters, batch_frac=batch_frac)

    # initialised microstates
    x_star_assim = refinement_sequence[end]

    x_R    = compute_present_state(system, x_R_assim, observations)
    x_star = compute_present_state(system, x_star_assim, observations)

    # initialised data
    y_star = integrate(system, x_star_assim, length(observations)).macrostates

    results_dict = Dict(
        "initial_guess" => x0,
        "rough_state_assim" => x_R_assim,
        "initialised_state_assim" => x_star_assim,
        "rough_state" => x_R,
        "initialised_state" => x_star,
        "fitted_observations" => y_star,
        "preprocessed_observations" => observations,
        "rough_costs" => rough_costs,
        "refinement_costs" => refinement_costs,
    )
    # results = [x0, x_R, x_star, y_star, observations, rough_costs, refinement_costs]
    
    if save_sequences
        sequences = Dict(
            "rough_sequence" => rough_sequence,
            "refinement_sequence" => refinement_sequence
        )
        merge!(results_dict, sequences)
    end
        
    return results_dict
end

initialise(system, x0, observations::Array{U}; kwargs...) where U<:Real = return initialise(system, x0, Data(observations); kwargs...)

# fixed number of refinement iters
function initialise(
    system::MicroMacroSystem, 
    x0::Array{P}, 
    observations::Data,
    num_iters_refine::Int64;
    α_R::Real,
    β_R::Real, 
    noise_scaling::Real=0, 
    q::Int64=0,
    method::Type{U}=Adam, 
    learning_rate=0.001, 
    batch_frac=1, 
    print_::Bool=false, 
    save_sequences::Bool=false,
    kwargs... 
    ) where {P <: Real, U <: Optimiser}

    ## PREPROCESS OBSERVATIONS!
    observations = moving_average(observations, q)

    ## BOUND SEARCH!
    x_R_assim, rough_num_steps, rough_costs, rough_sequence = initial_guess_search(system, x0, observations; 
            α_R=α_R, β_R=β_R, noise_scaling=noise_scaling, print_=print_)
    
    ## REFINE MICROSTATES!
    refinement_sequence, refinement_costs = cost_minimization(system, x_R_assim, observations, num_iters_refine;
            print_=print_, method=method,
            learning_rate=learning_rate, batch_frac=batch_frac)

    # initialised microstates
    x_star_assim = refinement_sequence[end]

    x_R    = compute_present_state(system, x_R_assim, observations)
    x_star = compute_present_state(system, x_star_assim, observations)

    # initialised data
    y_star = integrate(system, x_star_assim, length(observations)).macrostates

    results_dict = Dict(
        "initial_guess" => x0,
        "rough_state_assim" => x_R_assim,
        "initialised_state_assim" => x_star_assim,
        "rough_state" => x_R,
        "initialised_state" => x_star,
        "fitted_observations" => y_star,
        "preprocessed_observations" => observations,
        "rough_costs" => rough_costs,
        "refinement_costs" => refinement_costs,
    )
    # results = [x0, x_R, x_star, y_star, observations, rough_costs, refinement_costs]
    
    if save_sequences
        sequences = Dict(
            "rough_sequence" => rough_sequence,
            "refinement_sequence" => refinement_sequence
        )
        merge!(results_dict, sequences)
    end
        
    return results_dict
end
initialise(system, x0, observations::Array{U}, num_iters_refine; kwargs...) where U<:Real = return initialise(system, x0, Data(observations), num_iters_refine; kwargs...)