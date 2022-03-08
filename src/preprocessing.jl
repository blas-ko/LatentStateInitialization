############################## NOISE FILTERING FUNCTIONS #########################################

# moving average filter
function moving_average!(x::D) where D <: Dynamics

    x.macrostates[1] = 0.75*x.macrostates[1] + 0.25*x.macrostates[2]
    for i in 2:length(x.macrostates)-1
        x.macrostates[i] = 0.5x.macrostates[i]  + 0.25(x.macrostates[i-1] + x.macrostates[i+1])
    end
    x.macrostates[end] = 0.75x.macrostates[end]  + 0.25*x.macrostates[end-1]
    nothing
end

# iteration of the filter; it gets smoother for each iteration
function moving_average(x::D, q::Int64=1) where D <: Dynamics

    # Bring the option of doing nothing
    if q == 0
        return x

    else

        _x = deepcopy(x)
        for i in 1:q
          moving_average!(_x)
        end

        return _x
    end

    nothing
end

function SNR(signal, noisy)
    # P = ( 1/T ∑_t sₜ² )^1/2 → SNR = P_signal/P_noise
    return norm(signal)/norm(signal - noisy)
end

SNR(signal::Dynamics, noisy::Dynamics) = SNR(signal.macrostates, noisy.macrostates)
SNRdb(signal, noisy) = 10*log10(SNR(signal, noisy))