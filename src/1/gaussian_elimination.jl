function gaussian_elimination(
    A::AbstractMatrix{Tp},
    b::AbstractVector{Tp};
    atol::Tp = sqrt(eps(Tp))
) where {Tp <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    U = deepcopy(A)
    b = deepcopy(b)

    for k = 1:n - 1
        _, i = findmax(abs.(U[k:n, k]))
        i = i + k - 1
        @assert !isapprox(U[i, k], 0; atol=atol) "A must be a non-singular matrix"

        U[i, :], U[k, :] = U[k, :], U[i, :]
        b[i], b[k] = b[k], b[i]

        τ = vcat(zeros(Tp, k), U[k + 1:n, k] / U[k, k])
        e = vcat(zeros(Tp, k - 1), 1, zeros(Tp, n - k))

        M = I - τ * e'

        U = M * U
        b = M * b
    end

    c = upper_direct_substitution(U, b; atol=atol)

    return c
end
