using MosimoBase.LinearAlgebra: norm_sqr

function default_callback(setup::GDSetup)
    logging_period = max(10, setup.max_steps ÷ 20)
    final = original_positions(setup.final)
    @printf  " Step |  Distance  |  Length\n"
    (state::GDState; force_logging::Bool = false) -> begin
        if force_logging || state.step % logging_period ≡ 0
            @printf "%5d | %9.5f  | %8.5f" state.step √(state.dist²) state.path_length
        end
        final - positions(state.configuration)
    end
end

distance2(r1, r2) = norm_sqr(r1 - r2)
distance(r1, r2) = √(distance2(r1, r2))
distance(s1::MosiSystem, s2::MosiSystem) = distance(original_positions(s1), original_positions(s2))

function move!(
    rs, direction, model, E_L, δR
)
    ∇C = constraint_gradients(model, rs)
    V = potential_energy_function(model, rs)
    projector = if V > E_L
        ∇V = potential_energy_gradients(model, rs)
        projection_matrix([∇C..., ∇V])
    else
        projection_matrix(∇C)
    end
    T = eltype(direction)
    new_direction = unflatten(T, projector * flatten(direction))
    rs .+= δR * normalize(new_direction)
end

function constraint_corrector(C, ∇C)
    N_C = length(C)
    c = zeros(Float64, N_C)
    ∇C_unit_new = normalize.(∇C)
    for i in 1:N_C
        ∇Cᵢ_unit = copy(∇C_unit_new[i])
        F = Float64(1.0)
        c[i] = C[i] / norm(∇C[i])
        for j = 1:i-1
            γⱼᵢ = ∇C_unit_new[j] ⋅ ∇Cᵢ_unit
            F -= γⱼᵢ ^ 2
            c[i] += c[j] * γⱼᵢ
            ∇C_unit_new[i] -= ∇C_unit_new[j] * γⱼᵢ
        end
        ∇C_unit_new[i] /= √F
        c[i] /= -√F
    end
    sum(c .* ∇C_unit_new)
end

function constraint_corrections!(rs, model, E_L)
    N = length(rs)
    E_Lt = E_L + abs(E_L) * 1e-6
    δC = N * 1e-12
    C = constraints(model, rs)
    V = potential_energy_function(model, rs)
    constraint_satisfied = sum(C .^ 2) ≤ δC
    potential_satisfied = V ≤ E_Lt
    constraint_satisfied && potential_satisfied && return rs
    correction_iteration = 0
    while correction_iteration < 5 && !(constraint_satisfied && potential_satisfied)
        if any(isnan.(C)) || isnan(V)
            error("Something went wrong in the correction.")
        end
        ∇C = constraint_gradients(model, rs)
        if !potential_satisfied
            push!(C, V - E_L)
            ∇V = potential_energy_gradients(model, rs)
            push!(∇C, ∇V)
        end
        rs .+= constraint_corrector(C, ∇C)
        C = constraints(model, rs)
        V = potential_energy_function(model, rs)
        constraint_satisfied = sum(C .^ 2) ≤ δC
        potential_satisfied = V ≤ E_Lt
        correction_iteration += 1
    end
end

function gd_run(
    setup::GDSetup;
    callback::Nullable{Function} = nothing,
    return_result = true,
)
    model = setup.model
    δR = setup.δR
    max_steps = setup.max_steps

    state = init_state(setup)

    if isnothing(callback)
        callback = default_callback(setup)
    end

    @printf "GD started.\n"

    E_L = setup.landscape_energy
    current = positions(state.configuration)
    final = original_positions(setup.final)
    @printf "E_L: %8.5f\n" E_L
    @printf "Distance: %8.5f\n" distance(current, final)

    next_direction = callback(state)
    state.dist² = norm_sqr(next_direction)
    while true
        if state.dist² < δR ^ 2 || state.step ≥ max_steps
            break
        end
        previous = copy(current)
        move!(
            current, next_direction, model, E_L, δR
        )
        constraint_corrections!(current, model, E_L)
        state.step += 1
        state.path_length += distance(current, previous)
        next_direction = callback(state)
        state.dist² = norm_sqr(next_direction)
    end
    callback(state; force_logging = true)
    @printf "GD finished.\n"

    result = return_result ? GDResult(state.step, state.path_length, state.configuration) : state.path_length
end
