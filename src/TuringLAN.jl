module TuringLAN
    using Flux
    using Flux: logsumexp
    using Flux: params
    using ProgressMeter

    export train_model

    include("flux_utilities.jl")
end
