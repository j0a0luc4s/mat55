function upper_direct_substitution(A::Matrix{T}, b::Vector{T}; atol::T = 1e-6) where {T <: AbstractFloat}
	@assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
	n = size(A, 1)

	@assert all(all(isapprox.(A[i, 1:(i - 1)], 0, atol=atol)) for i = 1:n)  "A must be an upper triangular matrix"
	@assert all(!isapprox(A[i, i], 0, atol=atol) for i = 1:n) "A must be a non-singular matrix"

	c = zeros(length(b))

	for i = n:-1:1
		c[i] = (b[i] - A[i, :]' * c)/A[i, i]
	end

	return c
end
