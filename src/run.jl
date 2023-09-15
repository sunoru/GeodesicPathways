using MosimoBase.LinearAlgebra: norm_sqr

function default_callback(setup::GDSetup)
    logging_period = max(10, setup.max_steps ÷ 20)
    final = original_positions(setup.final)
    @printf " Step |  Distance  |  Length\n"
    (state::GDState; force_logging::Bool = false) -> begin
        if force_logging || state.step % logging_period ≡ 0
            @printf "%5d | %9.5f  | %8.5f\n" state.step √(state.dist²) state.path_length
        end
        final - positions(state.configuration)
    end
end

distance2(r1, r2) = norm_sqr(r1 - r2)
distance(r1, r2) = √(distance2(r1, r2))
distance(s1::MosiSystem, s2::MosiSystem) = distance(original_positions(s1), original_positions(s2))

function move!(
    rs, direction, model, E_L, δR,
    inplace_∇C
)
    ∇C = constraint_gradients(
        model, rs;
        inplace=@view(inplace_∇C[:, 1:end-1]),
        norm=true
    )
    V = potential_energy_function(model, rs)
    projector = if V > E_L
        ∇V = potential_energy_gradients(model, rs; inplace=@view(inplace_∇C[:, end]), norm=true)
        projection_matrix(inplace_∇C, normed=true)
    else
        projection_matrix(∇C, normed=true)
    end
    T = eltype(direction)
    new_direction = projector * direction
    rs .+= δR * normalize(new_direction)
end

# ∇C should not be normalized.
function constraint_corrector(
    C, ∇C,
)
    N_C = length(C)
    c = zeros(Float64, N_C)
    ∇C_unit_new = normalize_columns(∇C)
    @inbounds for i in 1:N_C
        ∇Cᵢ_unit = copy(∇C_unit_new[:, i])
        F = 1.0
        c[i] = C[i] / norm(∇C[:, i])
        for j = 1:i-1
            γⱼᵢ = ∇C_unit_new[:, j] ⋅ ∇Cᵢ_unit
            F -= γⱼᵢ^2
            c[i] += c[j] * γⱼᵢ
            ∇C_unit_new[:, i] -= ∇C_unit_new[:, j] * γⱼᵢ
        end
        ∇C_unit_new[:, i] /= √F
        c[i] /= -√F
    end
    sum(∇C_unit_new .* c', dims=2)
end

function constraint_corrections!(rs, model, E_L, inplace_C, inplace_∇C)
    N = length(rs)
    E_Lt = E_L + abs(E_L) * 1e-6
    δC = N * 1e-12
    C = constraints(model, rs; inplace=@view(inplace_C[1:end-1]))
    V = potential_energy_function(model, rs)
    constraint_satisfied = sum(C .^ 2) ≤ δC
    potential_satisfied = V ≤ E_Lt
    constraint_satisfied && potential_satisfied && return rs
    correction_iteration = 0
    while correction_iteration < 5 && !(constraint_satisfied && potential_satisfied)
        if any(isnan.(C)) || isnan(V)
            error("Something went wrong in the correction.")
        end
        ∇C = constraint_gradients(
            model, rs;
            inplace=@view(inplace_∇C[:, 1:end-1]),
            norm=false
        )
        if !potential_satisfied
            inplace_C[end] = V - E_L
            C = inplace_C
            ∇C = inplace_∇C
            ∇V = potential_energy_gradients(model, rs; inplace=@view(inplace_∇C[:, end]), norm=false)
        end
        rs .+= constraint_corrector(C, ∇C)
        C = constraints(model, rs; inplace=@view(inplace_C[1:end-1]))
        V = potential_energy_function(model, rs)
        constraint_satisfied = sum(C .^ 2) ≤ δC
        potential_satisfied = V ≤ E_Lt
        correction_iteration += 1
    end
end

function gd_run(
    setup::GDSetup,
    callback::Nullable{F}=nothing;
    return_result=true
) where {F<:Function}
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

    just_moved = zero_similar(current)
    # Only use one C or ∇C buffer and give them one extra column.
    inplace_C = constraints(model, current)
    push!(inplace_C, 0.0)
    inplace_∇C = constraint_gradients(model, current)
    inplace_∇C = [inplace_∇C zero_similar(current)]
    next_direction = callback(state)
    state.dist² = norm_sqr(next_direction)
    while true
        if state.dist² < δR^2 || state.step ≥ max_steps
            break
        end
        previous = copy(current)
        move!(
            current, next_direction, model, E_L, δR,
            inplace_∇C
        )
        constraint_corrections!(current, model, E_L, inplace_C, inplace_∇C)
        state.step += 1
        just_moved .= current .- previous
        state.path_length += norm(just_moved)
        next_direction = callback(state)
        state.dist² = norm_sqr(next_direction)
    end
    callback(state; force_logging=true)
    @printf "GD finished.\n"

    result = return_result ? GDResult(state.step, state.path_length, state.configuration) : state.path_length
end
