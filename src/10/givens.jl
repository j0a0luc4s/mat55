function givens(
    a::Tp,
    b::Tp;
    atol::Tp = 1e-6
) where {Tp <: AbstractFloat}
    if isapprox(b, 0.0; atol=atol)
        s = 0.0
        c = 1.0
    elseif abs(b) > abs(a)
        τ = -a/b
        s = 1.0/sqrt(1.0 + τ*τ)
        c = s*τ
    else
        τ = -b/a
        c = 1.0/sqrt(1.0 + τ*τ)
        s = c*τ
    end

    return [c  s;
            -s c]
end
