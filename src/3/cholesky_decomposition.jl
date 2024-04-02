function cholesky_decomposition(A::Matrix{T}; atol::T = 1e-6) where {T <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) "A must be a square matrix"

    n = size(A, 1)

    for i = 1:n
        M = A[1:i, 1:i]
        @assert det(M) > 0 && !isapprox(det(M), 0; atol=atol) "A must be positive definite"
    end

    G = zeros(n, n)

    @assert !isapprox(A[1, 1], 0; atol=atol) "Null Pivot"
    G[:, 1] = A[:, 1] / sqrt(A[1, 1])
    for i = 2:n
        G[i:n, i] = A[i:n, i]
        G[i:n, i] = G[i:n, i] - G[i:n, 1:(i-1)]*G[i, 1:(i-1)]
        @assert !isapprox(G[i, i], 0; atol=atol) "Null Pivot"
        G[i:n, i] = G[i:n, i] / sqrt(G[i, i])
    end

    return G
end
