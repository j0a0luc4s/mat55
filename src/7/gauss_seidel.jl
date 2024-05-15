function gauss_seidel(A::Matrix{T}, b::Vector{T}, x0::Vector{T}; atol::T = 1e-6, k::Int64 = 10000) where{T <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) == length(x0) "A, b and x0 dimension mismatch"
    n = size(A,1)

    @assert !isapprox(det(A), 0; atol=atol) "A must be a non-singular matrix"
    @assert !any(isapprox.(diag(A), 0; atol=atol)) "Null pivot"
    
    prev_x = deepcopy(x0)
    x = deepcopy(x0)
    count = 0
    while count < k
        for i = 1:n
            x[i] = (b[i] - dot(A[i, :], x) + A[i, i]*x[i])/A[i, i]
        end
        count += 1;
        if norm(x - prev_x)/norm(prev_x) < atol
            break;
        end
        prev_x = deepcopy(x)
    end

    return x, count
end
