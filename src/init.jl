function MosimoBase.init_state(setup::GDSetup)
    rng = new_rng(setup.seed)
    configuration = ConfigurationSystem(
        original_positions(setup.initial)
    )
    state = GDState(rng, 0, 0.0, configuration, ∞)
    state
end

function gd_init(
    initial,
    final;  # System or positions or (positions, periods)
    model::MosiModel,
    δR::Real = 0.05,
    landscape_energy::Nullable{Real} = nothing,
    δE::Real = 1e-8,
    max_steps::Integer = 100_000,
    seed::Nullable{Integer} = nothing
)
    box = pbc_box(model)
    initial = ConfigurationSystem(copy(initial); box = box, update_periods = false)
    final = ConfigurationSystem(copy(final); box = box, update_periods = false)
    N = natoms(initial)
    @assert N === natoms(final)
    E_L = max(potential_energy(initial, model), potential_energy(final, model))
    if landscape_energy === nothing
        landscape_energy = E_L
    elseif landscape_energy < E_L
        @warn "`landscape_energy` ($landscape_energy) is smaller than $(E_L)"
        # throw(SimulationError("`landscape_energy` should be at least $(E_L)."))
    end

    seed = isnothing(seed) ? make_seed() : make_seed(seed)

    setup = GDSetup(
        initial, final, N,
        δR, landscape_energy, δE,
        max_steps, seed, model
    )
end
