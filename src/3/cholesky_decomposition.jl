function cholesky_decomposition(
    A::AbstractMatrix{Tp};
    atol::Tp = sqrt(eps(Tp))
) where {Tp <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) "A must be a square matrix"
    n = size(A, 1)

    G = zeros(Tp, n, n)

    @assert !isapprox(sqrt(A[1, 1]), 0; atol=atol) "Null Pivot"
    G[:, 1] = A[:, 1] / sqrt(A[1, 1])
    for i = 2:n
        G[i:n, i] = A[i:n, i]
        G[i:n, i] = G[i:n, i] - G[i:n, 1:i - 1]*G[i, 1:i - 1]
        @assert !isapprox(sqrt(G[i, i]), 0; atol=atol) "Null Pivot"
        G[i:n, i] = G[i:n, i] / sqrt(G[i, i])
    end

    return G
end
