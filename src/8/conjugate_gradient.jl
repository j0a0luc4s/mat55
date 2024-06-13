function conjugate_gradient(A::Matrix{T}, b::Vector{T}, x0::Vector{T}; atol::T = 1e-8, kmax::Int64 = 10000) where{T <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) == length(x0) "A and b dimension mismatch"
    n = size(A,1)

    @assert all(det(A[1:i, 1:i]) > 0 && !isapprox(det(A[1:i, 1:i]), 0; atol) for i in 1:n) "A must be positive definite"

    prev_x = deepcopy(x0)
    x = deepcopy(x0)
    r = b - A*x
    d = deepcopy(r)
    count = 0
    while count < kmax
        Ad = A*d
        dtAd = d'*Ad

        α = r'*d/(dtAd)
        x = x + α*d
        r = b - A*x
        β = -r'*Ad/(dtAd)
        d = r + β*d

        count += 1
        if norm(x - prev_x)/norm(prev_x) < atol
            break
        end
        prev_x = deepcopy(x)
    end

    return x, count
end
