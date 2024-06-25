function lu_decomposition_nopivot(
    A::AbstractMatrix{Tp};
    atol::Tp = sqrt(eps(Tp))
) where {Tp <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) "A must be a square matrix"
    n = size(A, 1)

    L = Matrix{Tp}(I, n, n)
    U = deepcopy(A)

    for k = 1:n - 1
        @assert !isapprox(U[k, k], 0; atol=atol) "A must be a non-singular matrix"

        τ = vcat(zeros(Tp, k), U[k + 1:n, k] / U[k, k])
        e = vcat(zeros(Tp, k - 1), 1, zeros(Tp, n - k))

        M = I - τ * e'

        U = M * U
        L = L - M + I
    end

    return L, U
end
