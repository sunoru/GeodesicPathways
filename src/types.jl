struct GDSetup{M <: MosiModel} <: SimulationSetup
    initial::ConfigurationSystem
    final::ConfigurationSystem
    N::Int

    δR::Float64
    landscape_energy::Float64
    δE::Float64

    max_steps::Int
    seed::UInt64

    model::M
end
MosimoBase.mosi_model(setup::GDSetup) = setup.model

mutable struct GDState{R <: AbstractRNG} <: SimulationState
    rng::R
    step::Int
    path_length::Float64
    configuration::ConfigurationSystem
    dist²::Float64
end
MosimoBase.system(state::GDState) = state.configuration
MosimoBase.time(state::GDState) = state.step
path_length(state::GDState) = state.path_length

struct GDResult <: SimulationResult
    num_steps::Int
    path_length::Float64
    last_configuration::ConfigurationSystem
end

steps(result::GDResult) = result.num_steps
path_length(result::GDResult) = result.path_length
MosimoBase.system(result::GDResult) = result.last_configuration
