function lu_decomposition_nopivot(A::Matrix{T}; atol::T=1e-6) where {T<:AbstractFloat}
        @assert size(A, 1) == size(A, 2) "A must be a square matrix"
        n = size(A, 1)

        @assert !isapprox(det(A), 0; atol=atol) "A must be a non-singular matrix"

        L = Matrix{T}(I, n, n)
        U = deepcopy(A)

        for k = 1:(n-1)
                τ = vcat(zeros(k), U[(k+1):n, k] / U[k, k])
                e = vcat(zeros(k - 1), 1, zeros(n - k))

                M = I - τ * e'

                U = M * U
                L = L - M + I
        end

        return L, U
end
