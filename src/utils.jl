# get_configuration: i -> configurationᵢ
function available_configurations(get_configuration::Function, range, E_L::Float64, model::MosiModel)
    results = Int[]
    for i in range
        s = get_configuration(i)
        if potential_energy(s, model) ≤ E_L
            push!(results, i)
        end
    end
    results
end
