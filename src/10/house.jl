function house(
    x::Vector{Tp};
    atol::Tp = 1e-6
) where {Tp <: AbstractFloat}
    m = length(x)
    σ = x[2:m]'*x[2:m]
    v = vcat(1, x[2:m])
    
    if isapprox(σ, 0.0; atol=atol)
        β = 2.0
    else
        v[1] = -σ/(x[1] + sqrt(x[1]*x[1] + σ))
        β = 2*v[1]*v[1]/(σ + v[1]*v[1])
        v = v/v[1]
    end

    return v, β
end
