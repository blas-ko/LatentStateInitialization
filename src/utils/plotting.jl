using Plots, LaTeXStrings, StatsPlots
gr()
PLOT_ATTR = ( guidefont=font(16), tickfont=font(12), legendfont=font(10)  ) # figsize?

# ### PLOT ATTRIBUTES ###
default(fmt = :png)

# Defining some colors!!
PALETTE = (
    c_1 = RGBA(0.5,0.4,0.7,1),    # purple
    c_2 = RGBA(0.9, 0.4, 0.4),    # pink/red
    c_3 = RGBA(0.2,0.5,1,1),      # blue
    c_4 = RGBA(0.2, 0.7, 0.5, 1), # green (blue)
    # c_5 = RGBA(0.3,0.7,0.1,1),    # green (yellow)
    c_5 = RGBA(0.6,0.8,0.0,1),    # green (yellow)
    c_6 = RGBA(0.7,0.2,0.3,1),    # red
)

# past plotting functions 

function plot_microstate(
    x0,
    refinement_sequence,
    args...;
    kwargs...)

    pyplot() # we use pyplot because gr has problems with LaTexStrings

    x_rough = refinement_sequence[1]
    x_star  = refinement_sequence[end]

    @assert length(x_rough) == length(x0)
 
    # number of intermediate convergence
    num_intermediate_vals = 50
    dn = Int( round( ( length(refinement_sequence))/num_intermediate_vals + 1 ) )

    # define colormaps
    cmap = cgrad([PALETTE.c_4, PALETTE.c_5])
    # define plot labels
    lab_x_rough = L"\mathbf{x}_R" # "rough-guess microstate"
    lab_x_star  = L"\mathbf{x}^\star" # "initialised microstate"
    lab_x_0     = L"\mathbf{x}_0" # "ground-truth microstate"

    ## Plot microstate components
    scatter(refinement_sequence[1:dn:end]; m=:o, ms=5, msw=0.1, PALETTE=cmap, lab="", alpha=0.5)
    scatter!(x_rough; m=:o, ms=8, msw=0.7, c=PALETTE.c_5, lab=lab_x_rough, alpha=0.6)
    scatter!(x_star;  m=:o, ms=11, msw=1, c=PALETTE.c_4, lab=lab_x_star)
    scatter!(x0;      m=:X, ms=9, c=:black, lab=lab_x_0, alpha=0.8)

    ## Plot attributes
    
    # number of ticks shown in x
    N = length(x0)
    num_xticks = min( N, 10 )
    tick_size = Int( round(N/num_xticks) )
    x_ticks   = range(1, step=tick_size, length=num_xticks)
    x_strings = [latexstring("\\mathbf{x}_{$n}") for n in (x_ticks)]
    # y-limits
    ymin, ymax = get_ax_lims( x0, x_rough, x_star )

    # plot attributes
    xticks!(x_ticks, x_strings)
    ylims!(ymin, ymax)
    p = plot!(;PLOT_ATTR..., kwargs...)

    gr() # return to gr backend
    return p
end
# helper
function plot_microstate(initialisation_results::Dict, args...; kwargs...) 
    x0, refinement_sequence = initialisation_results["ground_truth_state"], initialisation_results["refinement_sequence"]
    return plot_microstate(x0, refinement_sequence, args...; kwargs...)
end

function plot_timeseries(
    system,
    x_star,
    x0,
    data,
    fitted_data,
    prediction_steps,
    args...;
    kwargs...) 

    T = length(data)

    predictions = integrate(system, x_star, T+prediction_steps).macrostates[T:end]
    projections = integrate(system, x0,     T+prediction_steps).macrostates[T:end]

    # Predefine labels
    lab_fit  = "inferred observations"
    lab_obs  = "ground truth observations"
    lab_pred = "predictions"
    # lab_proj = "perfect forecast"

    # Set axis attributes
    assim_window = (-T+1):0
    pred_window  = 0:prediction_steps

    ## Plotting 
    # division between assimilation and prediciton windows
    vline([0], lw=2, c=:grey, lab="") 

    plot!( assim_window, data,         lw=5, c=:black, lab=lab_obs)
    plot!( pred_window,   projections, lw=5, c=:black, lab="")
    plot!( assim_window, fitted_data,  lw=5, ls=:dash, c=PALETTE.c_2, lab=lab_fit)
    plot!( pred_window,   predictions, lw=5, ls=:dash, c=PALETTE.c_3, lab=lab_pred)
    

    xlabel!(L"k")
    ylabel!(L"y_k")

    p = plot!(; PLOT_ATTR..., kwargs...)

    return p

end
# helper
function plot_timeseries(system, initialisation_results::Dict, prediction_steps=nothing, args...; kwargs...) 
    x0 = initialisation_results["ground_truth_state"] # _assim 
    x_star = initialisation_results["initialised_state_assim"]
    data = initialisation_results["clean_observations"] + initialisation_results["noise"]
    fitted_data = initialisation_results["fitted_observations"] 
    if prediction_steps == nothing
        prediction_steps = initialisation_results["prediction_steps"]
    end
    return plot_timeseries(system, x_star, x0, data, fitted_data, prediction_steps, args...; kwargs...)
end

function plot_average_errors(
    errors_obs_space::Vector,
    errors_mod_space::Vector,
    errors_obs_space_noisy::Vector,
    errors_mod_space_noisy::Vector,
    T::Int64=0,
    num_samples::Real=Inf,
    kmax::Real=Inf,
    kmax_noisy::Real=Inf,
    lyap_time::Real=Inf, # infinity is a real number, somewhow
    args...;
    kwargs...)
            
    # sampling interval
    m = Int( round(length(errors_mod_space)/length(errors_obs_space)) )
    # number of samples to be plotted
    num_samples = Int(min(num_samples, length(errors_obs_space)))
    
    x_mod = (-T+1):1/m:(num_samples-T)
    x_obs = (-T+1):(num_samples-T)
    
    y_mod = errors_mod_space[1:length(x_mod)]
    y_obs = errors_obs_space[1:num_samples]
    
    y_mod_noisy = errors_mod_space_noisy[1:length(x_mod)]
    y_obs_noisy = errors_obs_space_noisy[1:num_samples]
    
    plot(  x_mod, y_mod, lw=2.3, c=PALETTE.c_5, ls=:dot, lab="error model space (noiseless)" )
    plot!( x_obs, y_obs, lw=2.8, c=PALETTE.c_5, lab="error obs space (noiseless)" )
    
    plot!( x_mod, y_mod_noisy, lw=2.3, c=PALETTE.c_3, ls=:dot, lab="error model space (noisy)" )
    plot!( x_obs, y_obs_noisy, lw=2.8, c=PALETTE.c_3, lab="error obs space (noisy)" )
    
    vline!([kmax], c=PALETTE.c_5, lab="") # L"k_{max}"
    vline!([kmax_noisy], c=PALETTE.c_3, lab="") # L"k_{max}^{noisy}"
    
    lab = ""
    if lyap_time != Inf
        lab = L"t_\lambda"
    end
    vline!( [lyap_time], c=:black, ls=:dash, lab=lab) # "lyap_time"
    vline!([0], c=:black, lab="", alpha=0.5)
    
    xlabel!(L"k")
    ylabel!(L"NSE_k")
    
    p = plot!(; PLOT_ATTR..., leg=:bottomright, yscale=:log10,kwargs...)
    
    return p
    
end

function plot_average_errors(
        ensemble_results_noiseless::Dict,
        ensemble_results_noisy::Dict,
        features::Dict,
        num_samples=Inf,
        args...;
        kwargs...
    )

    # Compute median errors
    errors_obs_space_median = median(ensemble_results_noiseless["errors_star_obs_space"])
    errors_mod_space_median = median(ensemble_results_noiseless["errors_star_model_space"])
    errors_obs_space_noisy_median = median(ensemble_results_noisy["errors_star_obs_space"])
    errors_mod_space_noisy_median = median(ensemble_results_noisy["errors_star_model_space"])
    
    # Compute mean errors
    errors_obs_space_mean = mean(ensemble_results_noiseless["errors_star_obs_space"])
    errors_mod_space_mean = mean(ensemble_results_noiseless["errors_star_model_space"])
    errors_obs_space_noisy_mean = mean(ensemble_results_noisy["errors_star_obs_space"])
    errors_mod_space_noisy_mean = mean(ensemble_results_noisy["errors_star_model_space"])

    lyap_time = features["lyapunov_time"]
    
    T = length(ensemble_results_noiseless["fitted_observations_array"][1])
    kmax       = mean( ensemble_results_noiseless["prediction_horizons_obs_space"] )
    kmax_noisy = mean( ensemble_results_noisy["prediction_horizons_obs_space"] )
    
    p_median = plot_average_errors(
        errors_obs_space_median,
        errors_mod_space_median, 
        errors_obs_space_noisy_median, 
        errors_mod_space_noisy_median,
        T,
        num_samples,
        args...; kwargs...)

    p_mean = plot_average_errors(
        errors_obs_space_mean, 
        errors_mod_space_mean, 
        errors_obs_space_noisy_mean, 
        errors_mod_space_noisy_mean,
        T,
        num_samples,
        kmax,
        kmax_noisy,
        lyap_time, 
        args...; kwargs...)
    
    return p_median, p_mean
end

function plot_varying_T(
    T_range::AbstractArray,
    y_vars::Vector{U};
    kwargs...) where U # {U <: Number}

    plot( T_range, y_vars; lw=2, lab="", kwargs... )
    xlabel!("Number of observations (T)")

    return plot!(; PLOT_ATTR..., kwargs... )
end

function plot_varying_T(
    varying_T_df::DataFrame,
    varying_T_df_noisy::DataFrame,
    args...;
    kwargs...)

    @assert varying_T_df.T == varying_T_df_noisy.T

    ## READING VARS 

    # noiseless quantities
    T_range = varying_T_df.T

    norm_x_xstar = varying_T_df.norm_x_xstar
    norm_x_xstar_std = varying_T_df.norm_x_xstar_std

    nowcast_error = varying_T_df.nowcast_error
    nowcast_error_std = varying_T_df.nowcast_error_std

    prediction_horizons = varying_T_df.prediction_horizon 
    prediction_horizons_std = varying_T_df.prediction_horizon_std

    # prediction_horizon = varying_T_df.prediction_horizon 
    
    suc_rate = varying_T_df.success_rate

    rough_lengths = varying_T_df.rough_length
    rough_lengths_std = varying_T_df.rough_length_std

    refinement_lengths = varying_T_df.refinement_length
    refinement_lengths_std = varying_T_df.refinement_length_std

    δr = varying_T_df.delta_r

    # noisy quantities
    T_range_noisy = varying_T_df_noisy.T

    norm_x_xstar_noisy = varying_T_df_noisy.norm_x_xstar
    norm_x_xstar_std_noisy = varying_T_df_noisy.norm_x_xstar_std

    nowcast_error_noisy = varying_T_df_noisy.nowcast_error
    nowcast_error_std_noisy = varying_T_df_noisy.nowcast_error_std

    prediction_horizons_noisy = varying_T_df_noisy.prediction_horizon
    prediction_horizons_std_noisy = varying_T_df_noisy.prediction_horizon_std
   
    suc_rate_noisy = varying_T_df_noisy.success_rate

    rough_lengths_noisy = varying_T_df_noisy.rough_length
    rough_lengths_std_noisy = varying_T_df_noisy.rough_length_std

    refinement_lengths_noisy = varying_T_df_noisy.refinement_length
    refinement_lengths_std_noisy = varying_T_df_noisy.refinement_length_std

    δr_noisy = varying_T_df_noisy.delta_r

    ## PLOTS 

    # # plot microstate_error
    # lab = "noiseless"
    # ylabel = "average microstate error"
    # p1 = plot_varying_T( T_range, norm_x_xstar;
    #         # ribbon=norm_x_xstar_std,
    #         lab=lab,
    #         ylabel=ylabel,
    #         yscale=:log10,
    #         m=:X, ms=5, c=:grey, 
    #         kwargs...)

    # # plot microstate_error noisy
    # lab = "noisy"
    # ylabel = "average microstate error"
    # p1n = plot_varying_T( T_range, norm_x_xstar_noisy;
    #         # ribbon=norm_x_xstar_noisy_std,
    #         lab=lab,
    #         ylabel=ylabel,
    #         yscale=:log10,
    #         m=:X, ms=5, c=:grey, 
    #         kwargs...)

    # Plot microstate error
    lab = ["noiseless" "noisy"]
    ylabel = "initialised microstate error"
    p1 = plot_varying_T( T_range, [norm_x_xstar, norm_x_xstar_noisy];
            lab=lab,
            ylabel=ylabel,
            yscale=:log10,
            m=:X, ms=5, c=[PALETTE.c_5 PALETTE.c_3],
            legend=:topright,
            kwargs...)

    # Plot nowcast error
    lab = ["noiseless" "noisy"]
    ylabel = "nowcast error"
    p2 = plot_varying_T( T_range, [nowcast_error, nowcast_error_noisy];
            lab=lab,
            ylabel=ylabel,
            yscale=:log10,
            m=:X, ms=5, c=[PALETTE.c_5 PALETTE.c_3],
            legend=:topright,
            kwargs...)


    # # plot nowcast_error
    # lab = "noiseless"
    # ylabel = "average nowcast error"
    # p2 = plot_varying_T( T_range, nowcast_error;
    #         # ribbon=nowcast_error_std,
    #         lab=lab,
    #         ylabel=ylabel,
    #         yscale=:log10,
    #         m=:d, ms=5, c=:grey, 
    #         kwargs...)

    # hline!( [sqrt(δr[1])], c=:black, lab=L"\sqrt{ \delta_r }" )
    hline!( [δr[1]], c=PALETTE.c_5, lab="")#, lab=L"\delta_r" )

    # plot nowcast error noisy 
    # lab = "noisy"
    # ylabel = "average nowcast error"
    # p2n = plot_varying_T( T_range, nowcast_error_noisy;
    #         # ribbon=nowcast_error_std_noisy,
    #         lab=lab,
    #         ylabel=ylabel,
    #         yscale=:log10,
    #         m=:d, ms=5, c=:grey,
    #         kwargs...)

    # hline!( [sqrt(δr_noisy[1])], c=:black, lab=L"\sqrt{ \delta_r }" )
    hline!( [δr_noisy[1]], c=PALETTE.c_3, lab="")#, lab=L"\delta_r" )


    # Plot prediction horizons
    lab = ["noiseless" "noisy"]
    ylabel = "prediction horizon"
    p3 = plot_varying_T( T_range, [prediction_horizons, prediction_horizons_noisy];
            lab=lab,
            ylabel=ylabel,
            m=:d, ms=5, c=[PALETTE.c_5 PALETTE.c_3],
            legend=:topleft,
            kwargs...)

    # hline!([lyap_time], c=:black, alpha=0.6, lab=L"t_\lambda") # "10-fold time"

    # Plot prediction horizons
    # lab = ["noiseless" "noisy"]
    # ylabel = "prediction horizon (shitty one)"
    # p3m = plot_varying_T( T_range, [prediction_horizons, prediction_horizons_noisy];
    #         # ribbon=[prediction_horizon_std, prediction_horizon_std_noisy],
    #         lab=lab,
    #         ylabel=ylabel,
    #         m=:d, ms=5, c=[PALETTE.c_4 PALETTE.c_5],
    #         kwargs...)

    # hline!([lyap_time], c=:black, alpha=0.6, lab=L"t_\lambda") # "10-fold time"

    # Plot success rates
    lab = ["noiseless" "noisy"]
    ylabel = "success rate"
    p4 = plot_varying_T( T_range, [suc_rate, suc_rate_noisy];
            lab=lab,
            ylabel=ylabel,
            m=:o, ms=5, c=[PALETTE.c_6 PALETTE.c_2],
            kwargs...)

    # Plot rough lengths
    lab = ["noiseless" "noisy"]
    ylabel = "num rough steps"
    p5 = plot_varying_T( T_range, [rough_lengths, rough_lengths_noisy];
            # ribbon=[rough_lengths_std, rough_lengths_std_noisy],
            lab=lab,
            ylabel=ylabel,
            yscale=:log10,
            m=:o, ms=5, c=[PALETTE.c_5 PALETTE.c_3],
            kwargs...)

    # Plot refinement_cost lengths
    lab = ["noiseless" "noisy"]
    ylabel = "num epochs"
    p6 = plot_varying_T( T_range, [refinement_lengths, refinement_lengths_noisy];
            # ribbon=[refinement_lengths_std, refinement_lengths_std_noisy],
            lab=lab,
            ylabel=ylabel,
            yscale=:log10,
            m=:o, ms=5, c=[PALETTE.c_5 PALETTE.c_3],
            kwargs...)

    return p1,p2,p3,p4,p5,p6
    
end

function plot_varying_T_and_m(
    T_range::AbstractArray,
    m_range::AbstractArray,
    z_val::Matrix{U},
    N = nothing;
    kwargs...) where U

    cmap = cgrad([:blue, :white, :red])

    heatmap(T_range, m_range, z_val, c=cmap; kwargs...)
    if N != nothing
        plot!(N ./ m_range, m_range, m=:o, lw=3, ms=7, c=:white, lab=L"T_c")
    end
    xticks!(T_range)
    yticks!(m_range)

    #alternative
    # z = log10.( C2["nowcast_error_matrix"] )
    # lab = reshape("m = " .* string.(m_range), 1,length(m_range) )
    # plot(T_range, z_val', m=:o, ms=3, lab=lab )
    # vline!(N ./ m_range, c=:black, alpha=0.3, lab=L"T_c")
    # hline!([log10(sqrt(delta_r))], c=:black, lab="")

    xlabel!("number of observations (T)")
    ylabel!("sampling interval (m)")

    return plot!(; PLOT_ATTR..., kwargs...)

end

function plot_varying_T_and_m(
    results_varying_T_and_m_dict::Dict;
    kwargs...) where U

    T_range = results_varying_T_and_m_dict["num_observations_array"]
    m_range = results_varying_T_and_m_dict["sampling_interval_array"]
    # N       = results_varying_T_and_m_dict["microstate_dimension"]

    # plot nowcast error matrix  
    z = log10.( results_varying_T_and_m_dict["nowcast_error_matrix"] )
    title = "nowcast error"
    p1 = plot_varying_T_and_m( T_range, m_range, z;
            # N,
            title=title,
            kwargs... )

    # plot nowcast error model space matrix
    z = log10.( results_varying_T_and_m_dict["microstate_error_matrix"] ) 
    title = "microstate error"
    p2 = plot_varying_T_and_m( T_range, m_range, z;
            # N,
            title=title,
            kwargs... )



    # plot success_rate matrix
    z = results_varying_T_and_m_dict["success_rate_matrix"]
    title = "success rate"
    cmap = cgrad([:red, :green]) 
    p3 = plot_varying_T_and_m( T_range, m_range, z;
    # N,
    c=cmap,
    title=title,
    kwargs... )

    # plot prediction horizon matrix
    z = results_varying_T_and_m_dict["prediction_horizon_matrix"]  .* m_range
    title="normalised prediction horizon"
    cmap = cgrad([:blue, :white, :red]) 
    p4 = plot_varying_T_and_m( T_range, m_range, z;
    # N,
    title=title,
    c=cmap,
    kwargs... )

    # # plot prediction horizon median matrix
    # z = results_varying_T_and_m_dict["prediction_horizon_mean_matrix"] .* m_range
    # title="normalised prediction horizon (shitty one)"
    # p3m = plot_varying_T_and_m( T_range, m_range, z;
    # # N,
    # title=title,
    # c=cmap,
    # kwargs... )

    return p1,p2,p3,p4

end

function plot_varying_rough_threshold(
    x_variables::Vector{Vector{U}}, 
    y_variables::Vector{Vector{V}},
    delta_Rs::Vector,
    color_grad="Blues",
    args...;
    kwargs...) where {U <: Number, V <: Number}
    
    # Create plot labels
    deltas = round.(delta_Rs, sigdigits=1)
    num_deltas = length( deltas )
    labels_deltaR = "\\delta_R = " .* string.( deltas )
    labels_deltaR = reshape(labels_deltaR, 1, num_deltas)

    colors = colormap(color_grad, num_deltas)'

    scatter( x_variables, y_variables, c=colors, lab=labels_deltaR )

    plot!( scale=:log10; PLOT_ATTR..., kwargs... )

end 

function plot_varying_rough_threshold(
    results_rough_thresholds_dict::Dict,
    args...;
    kwargs...)

    dxRa    = results_rough_thresholds_dict["norms_x_xR_assim"]
    dxstara = results_rough_thresholds_dict["norms_x_xstar_assim"]
    dxR     = results_rough_thresholds_dict["norms_x_xR"]
    dxstar  = results_rough_thresholds_dict["norms_x_xstar"]
    dyR     = results_rough_thresholds_dict["norms_y_yR"]
    dystar  = results_rough_thresholds_dict["norms_y_ystar"]

    deltas   = results_rough_thresholds_dict["delta_R"]
    lengthsR = results_rough_thresholds_dict["rough_lengths"]
    lengthsr = results_rough_thresholds_dict["refinement_lengths"]
    suc_rate = results_rough_thresholds_dict["success_rates"]

    # xR vs xstar 
    p1 = plot_varying_rough_threshold( dxR, dxstar, deltas;
        leg=:topleft, 
        xlabel=L"|| \mathbf{x}_0 - \mathbf{x}_R ||_2",
        ylabel=L"|| \mathbf{x}_0 - \mathbf{x}_\star ||_2",
        kwargs... 
    )

    ax_min, ax_max = get_ax_lims( dxR, dxstar )
    plot!( [ax_min, ax_max], [ax_min, ax_max], c=:black, lw=2, lab="" )
    ylims!(ax_min, ax_max)
    xlims!(ax_min, ax_max)

    # xR vs xstar assimilative
    p1a = plot_varying_rough_threshold( dxRa, dxstara, deltas;
    leg=:topleft, 
    xlabel=L"|| \mathbf{x}_{-T} - \mathbf{x}_R^a ||_2",
    ylabel=L"|| \mathbf{x}_{-T} - \mathbf{x}_\star^a ||_2",
    kwargs... 
    )
    
    ax_min, ax_max = get_ax_lims( dxRa, dxstara )
    plot!( [ax_min, ax_max], [ax_min, ax_max], c=:black, lw=2, lab="" )
    ylims!(ax_min, ax_max)
    xlims!(ax_min, ax_max)
    
    # xR vs rough lengths 
    p2 = plot_varying_rough_threshold( dxR, lengthsR, deltas;
        leg=:topright, 
        xlabel=L"|| \mathbf{x}_0 - \mathbf{x}_R ||_2",
        ylabel="num rough steps",
        kwargs... 
    )
    
    # xR vs refinement length
    plot() 
    p3 = plot_varying_rough_threshold( dxR, lengthsr, deltas;
        leg=:bottomright, 
        xlabel=L"|| \mathbf{x}_0 - \mathbf{x}_R ||_2",
        ylabel="num epochs",
        kwargs... 
    )

    # yR vs xR
    plot()
    p4 = plot_varying_rough_threshold( dyR, dxR, deltas;
        leg=:topleft, 
        ylabel=L"|| \mathbf{x}_0 - \mathbf{x}_R ||_2",
        xlabel=L"|| \mathbf{y} - \mathbf{y}_R ||_2",
        kwargs... 
    )
    # plot!(median.( dyR ), sqrt.( deltas ),  m=:o, c=:red, lab=L"\sqrt{ \delta_R }", leg=:topleft)

    # yR vs xstar
    plot()
    p5 = plot_varying_rough_threshold( dyR, dxstar, deltas;
        leg=:topleft, 
        ylabel=L"|| \mathbf{x}_0 - \mathbf{x}_\star ||_2",
        xlabel=L"|| \mathbf{y} - \mathbf{y}_R ||_2",
        kwargs... 
    )
    # plot!( median.( dyR ), sqrt.( deltas ),  m=:o, c=:red, lab=L"\sqrt{ \delta_R }", leg=:topleft)

    plot()
    p6 = plot( deltas, suc_rate, m=:o, lw=2.5, lab="" )
    plot!(;
        ylabel="success rate", 
        xlabel=L"\delta_R",
        xscale=:log10,
        PLOT_ATTR...,
        kwargs... 
    )

    return (p1,p1a,p2,p3,p4,p5,p6)
end

function plot_varying_optimisers(
    optimiser_names,
    y_variable,
    y_std=0;
    label::String="",
    kwargs...)
    
    bar(optimiser_names, y_variable, yerror=y_std, xrotation=45, lab=""; kwargs...) #, label=label)
    
    ylabel!(label)
    
end

function plot_varying_optimisers(
    optimiser_names,
    y1_variable,
    y2_variable,
    y1_std,
    y2_std;
    label::String="",
    kwargs...)

    @assert length(y1_variable) == length(y2_variable)

    num_optimisers = length(optimiser_names)
     
    x_names = repeat(optimiser_names, outer = 2)
    ys =      hcat(y1_variable, y2_variable)
    ys_std =  hcat(y1_std, y2_std)
    ctg =     repeat(["Noiseless", "Noisy"], inner = length(optimiser_names)) 

    groupedbar(x_names, ys, yerror=ys_std, group = ctg, alpha=[1 0.4], bar_width=0.8, xrotation=45; kwargs... )
    ylabel!(label)
    
end

function plot_varying_optimisers(
    results_optimisers_df::DataFrame;
    kwargs...)
    
    # Obtain variables from dictionaries
    optimisers = results_optimisers_df.optimiser
    
    std_ratio = 1/5
    
    microstate_error = results_optimisers_df.microstate_error
    microstate_error_std = results_optimisers_df.microstate_error_std * std_ratio
    label_me = "microstate error" 

    microstate_assim_error = results_optimisers_df.microstate_assim_error
    microstate_assim_error_std = results_optimisers_df.microstate_assim_error_std * std_ratio
    label_mae = "assimilative microstate error" 
    
    refinement_cost = results_optimisers_df.refinement_cost
    refinement_cost_std = results_optimisers_df.refinement_cost_std * std_ratio
    label_cost = "cost"
    
    refinement_length =  results_optimisers_df.refinement_length
    refinement_length_std = results_optimisers_df.refinement_length_std * std_ratio
    label_length = "number of iterations"
    
    suc_rate = results_optimisers_df.success_rate
    label_sr = "succes rate"
    
    # Plot everything
    p1 = plot_varying_optimisers(
        optimisers,
        microstate_error,
        microstate_error_std;
        label = label_me,
        c = PALETTE.c_1,
        yscale=:log10,
        kwargs...
    )

    p1a = plot_varying_optimisers(
        optimisers,
        microstate_assim_error,
        microstate_assim_error_std;
        label = label_mae,
        c = PALETTE.c_2,
        yscale=:log10,
        kwargs...
    )
    
    p2 = plot_varying_optimisers(
        optimisers,
        refinement_cost,
        refinement_cost_std;
        label = label_cost,
        c = :pink,
        yscale = :log10,
        kwargs...
    )
    
    p3 = plot_varying_optimisers(
        optimisers,
        refinement_length,
        refinement_length_std/std_ratio;
        label = label_length,
        c = PALETTE.c_3,
        kwargs...
    )
    
    p4 = plot_varying_optimisers(
        optimisers,
        suc_rate,
        0;
        label = label_sr,
        c = PALETTE.c_6,
        kwargs...
    )
    
    return p1, p1a, p2, p3, p4
end 

function plot_varying_optimisers(
    results_optimisers_df::DataFrame,
    results_optimisers_df_noisy::DataFrame;
    kwargs...)
    
    # Obtain variables from dictionaries
    optimisers = results_optimisers_df.optimiser
    
    std_ratio = 1/5
    
    # Variables for noiseless initialisations
    microstate_error = results_optimisers_df.microstate_error
    microstate_error_std = results_optimisers_df.microstate_error_std
    label_me = "microstate error" 

    microstate_assim_error = results_optimisers_df.microstate_assim_error
    microstate_assim_error_std = results_optimisers_df.microstate_assim_error_std
    label_mae = "assimilative microstate error" 
    
    refinement_cost = results_optimisers_df.refinement_cost
    refinement_cost_std = results_optimisers_df.refinement_cost_std
    label_cost = "cost"
    
    refinement_length =  results_optimisers_df.refinement_length
    refinement_length_std = results_optimisers_df.refinement_length_std
    label_length = "number of epochs"
    
    suc_rate = results_optimisers_df.success_rate
    label_sr = "succes rate"

    # Variable for noisy initialisations
    microstate_error_noisy = results_optimisers_df_noisy.microstate_error
    microstate_error_std_noisy = results_optimisers_df_noisy.microstate_error_std

    microstate_assim_error_noisy = results_optimisers_df_noisy.microstate_assim_error
    microstate_assim_error_std_noisy = results_optimisers_df_noisy.microstate_assim_error_std
    
    refinement_cost_noisy = results_optimisers_df_noisy.refinement_cost
    refinement_cost_std_noisy = results_optimisers_df.refinement_cost_std
    
    refinement_length_noisy =  results_optimisers_df_noisy.refinement_length
    refinement_length_std_noisy = results_optimisers_df_noisy.refinement_length_std 
    
    suc_rate_noisy = results_optimisers_df_noisy.success_rate
    
    # Plot everything
    p1 = plot_varying_optimisers(
        optimisers,
        microstate_error,
        microstate_error_noisy,
        0, #microstate_error_std * std_ratio,
        0, #microstate_error_std_noisy * std_ratio;
        label = label_me,
        c = PALETTE.c_1,
        yscale = :log10,
        kwargs...
    )

    p1a = plot_varying_optimisers(
        optimisers,
        microstate_assim_error,
        microstate_assim_error_noisy,
        0, #microstate_assim_error_std * std_ratio,
        0, #microstate_assim_error_std_noisy * std_ratio;
        label = label_mae,
        c = :pink,
        yscale = :log10,
        kwargs...
    )
    
    p2 = plot_varying_optimisers(
        optimisers,
        refinement_cost,
        refinement_cost_noisy,
        0, #refinement_cost_std * std_ratio,
        0, #refinement_cost_std_noisy * std_ratio/3;
        label = label_cost,
        c = PALETTE.c_2,
        yscale = :log10,
        kwargs...
    )
    
    p3 = plot_varying_optimisers(
        optimisers,
        refinement_length,
        refinement_length_noisy,
        0, #refinement_length_std,
        0, #refinement_length_std_noisy;
        label = label_length,
        c = PALETTE.c_3,
        yscale = :log10,
        leg=:bottomright,
        kwargs...
    )
    
    p4 = plot_varying_optimisers(
        optimisers,
        suc_rate,
        suc_rate_noisy,
        0,
        0;
        label = label_sr,
        c = PALETTE.c_6,
        ylims=(0.7, 1.01),
        leg=:bottomleft,
        kwargs...
    )
    
    return p1, p1a, p2, p3, p4
end 

function plot_optimisers_behaviour(
    behaviours_df::DataFrame,
    delta_r::Real=Inf,
    std_ratio::Real=1/8,
    args...;
    kwargs...)

    optimisers_names = names(behaviours_df)[1:2:end]

    # initialise plot
    plot()
        
    # plot each optimiser behaviour independently
    for (i, optimiser) in enumerate(optimisers_names)
        
        behaviour = behaviours_df[!, optimiser]
        behaviour_std = behaviours_df[!, optimiser*"_std"] * std_ratio
        colour = PALETTE[ mod1(i, length(PALETTE)) ]
        
        plot!(behaviour, lw=3, lab=optimiser, ribbon=behaviour_std, c=colour)
    end

    hline!([delta_r], c=:black, lab=L"\delta_r")

    xlabel!("epoch")
    ylabel!("cost")

    p = plot!(yscale=:log10, leg=:topright; PLOT_ATTR..., kwargs...)

    return p
end

function plot_optimisers_behaviour(
    behaviours_df::DataFrame,
    ensemble_results::Dict,
    std_ratio::Real=1/8,
    args...;
    kwargs...)

    delta_r = compute_threshold_parameters(; ensemble_results["initialisation_params"]... )[2]
    return plot_optimisers_behaviour(behaviours_df, delta_r, std_ratio, args...; kwargs...)
end

function plot_powerspectra(
    freqs::AbstractArray,
    powers::AbstractArray, 
    freq_ticks=2:2:12,
    args...;
    kwargs...)

    sampling_freqs = 1 ./ freq_ticks

    plot( freqs, powers, lw=2.5, lab="" )
    vline!( sampling_freqs, c=:black, alpha=0.15, lab="" )

    # plot frequencies as ratios
    xticks!(sampling_freqs, "1/" .* string.(freq_ticks))
    
    xlabel!("frequency")
    ylabel!("power")

    maxy = 1.4 * maximum( powers[2:end] )
    ylims!(1e-7, maxy)

    # xlims: (0, 0.2) for MG, yscale=:log10
    p = plot!(; PLOT_ATTR..., kwargs...)
    return p

end

function plot_powerspectra(features::Dict, freq_ticks=2:2:12, args...; kwargs...)

    freqs = features["power_spectra_all_samples_x"]
    powers = features["power_spectra_all_samples_y"]

    return plot_powerspectra(freqs, powers, freq_ticks, args...; kwargs...)
end

### HELPER FUNCS ### 

function get_ax_lims(args...; ratio::Float64=0.05, kwargs...)

    # ax_min = minimum( minimum.(args) )
    # ax_max = maximum( maximum.(args) )

    ax_min = minimum(minimum.(min.( args... )))
    ax_max = maximum(maximum.(max.( args... )))

    return (1-ratio)*ax_min, (1+ratio)*ax_max

end

function save_plots(
    plotnames::Tuple,
    path::String="./"; format="png")

    @assert eltype(plotnames) == Symbol "plot names should be passed as Symbols for them to be evaluated"

    for p in plotnames
        fn = string(p)
        fn = replace(fn, "p_" => "")
        fn *= "."*format
        fig = eval(p)
        
        savefig(fig, path*fn)
    end

    println("Saved $(length(plotnames)) figures in the directory $path")
    nothing
end

save_plots(path::String, args...; kwargs...) = save_plots(args, path; kwargs...)

macro varname_to_symbol(args...) 
    Symbol.(args)
end