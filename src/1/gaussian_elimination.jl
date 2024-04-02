function gaussian_elimination(A::Matrix{T}, b::Vector{T}; atol::T = 1e-6) where {T <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    @assert !isapprox(det(A), 0; atol=atol) "A must be a non-singular matrix"

    _A = deepcopy(A)
    _b = deepcopy(b)

    for k = 1:(n - 1)
        τ = vcat(zeros(k), _A[(k + 1):n, k] / _A[k, k])
        e = vcat(zeros(k - 1), 1, zeros(n - k))

        M = I - τ * e'

        _A = M * _A
        _b = M * _b
    end

    c = upper_direct_substitution(_A, _b)

    return c
end
