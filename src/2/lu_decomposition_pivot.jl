function lu_decomposition_pivot(
    A::AbstractMatrix{Tp};
    atol::Tp = sqrt(eps(Tp))
) where {Tp <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) "A must be a square matrix"
    n = size(A, 1)

    L = Matrix{Tp}(I, n, n)
    U = deepcopy(A)
    P = Matrix{Tp}(I, n, n)

    for k = 1:n - 1
        _, i = findmax(abs.(U[k:n, k]))
        i = i + k - 1
        @assert !isapprox(U[i, k], 0; atol=atol) "A must be a non-singular matrix"

        L[i, 1:k - 1], L[k, 1:k - 1] = L[k, 1:k - 1], L[i, 1:k - 1]
        U[i, :], U[k, :] = U[k, :], U[i, :]
        P[i, :], P[k, :] = P[k, :], P[i, :]


        τ = vcat(zeros(Tp, k), U[k + 1:n, k] / U[k, k])
        e = vcat(zeros(Tp, k - 1), 1, zeros(Tp, n - k))

        M = I - τ * e'

        U = M * U
        L = L - M + I
    end

    return L, U, P
end
