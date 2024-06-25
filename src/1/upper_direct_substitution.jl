function upper_direct_substitution(
    A::AbstractMatrix{Tp},
    b::AbstractVector{Tp};
    atol::Tp = sqrt(eps(Tp))
) where {Tp <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    @assert all(all(isapprox.(A[i, 1:i - 1], 0, atol=atol)) for i = 1:n)  "A must be an upper triangular matrix"
    @assert all(!isapprox(A[i, i], 0, atol=atol) for i = 1:n) "A must be a non-singular matrix"

    c = zeros(Tp, n)

    for i = n:-1:1
        c[i] = (b[i] - A[i, :]' * c)/A[i, i]
    end

    return c
end
