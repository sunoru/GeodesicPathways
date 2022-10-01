module GeodesicPathways

using MosimoBase
using Printf

include("./utils.jl")

export steps, path_length, distance
export GDSetup, GDState, GDResult
include("./types.jl")

export gd_init
include("./init.jl")
export gd_run
include("./run.jl")

end # module
