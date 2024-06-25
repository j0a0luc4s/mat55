using LinearAlgebra

function upper_direct_substitution(
    A::Matrix{T},
    b::Vector{T};
    atol::T=1e-6
) where {T<:AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    @assert all(all(isapprox.(A[i, 1:(i-1)], 0, atol=atol)) for i = 1:n) "A must be an upper triangular matrix"
    @assert all(!isapprox(A[i, i], 0, atol=atol) for i = 1:n) "A must be a non-singular matrix"

    c = zeros(length(b))

    for i = n:-1:1
        c[i] = (b[i] - A[i, :]' * c) / A[i, i]
    end

    return c
end

function lower_direct_substitution(
    A::Matrix{T},
    b::Vector{T};
    atol::T=1e-6
) where {T<:AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    @assert all(all(isapprox.(A[i, (i+1):n], 0, atol=atol)) for i = 1:n) "A must be a lower triangular matrix"
    @assert all(!isapprox(A[i, i], 0, atol=atol) for i = 1:n) "A must be a non-singular matrix"

    c = zeros(length(b))

    for i = 1:n
        c[i] = (b[i] - A[i, :]' * c) / A[i, i]
    end

    return c
end

function gaussian_elimination(
    A::Matrix{T},
    b::Vector{T};
    atol::T=1e-6
) where {T<:AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    _A = deepcopy(A)
    _b = deepcopy(b)

    for k = 1:(n-1)
        @assert !isapprox(_A[k, k], 0; atol=atol) "A must be a non-singular matrix"

        τ = vcat(zeros(k), _A[(k+1):n, k] / _A[k, k])
        e = vcat(zeros(k - 1), 1, zeros(n - k))

        M = I - τ * e'

        _A = M * _A
        _b = M * _b
    end

    c = upper_direct_substitution(_A, _b)

    return c
end

println("Type in the number of variables: ")
n = parse(Int64, readline())
println("Type in the square matrix A (", n, " lines with ", n, " numbers each): ")
A = Matrix{Float64}(undef, n, n)
for i = 1:n
    A[i, :] = Float64[parse(Float64, el) for el in split(readline())]
end
println("Type in the vector b (1 line with ", n, " numbers): ")
b = Vector{Float64}(undef, n)
b = Float64[parse(Float64, el) for el in split(readline())]
println("Select a method (type 1-3): ")
println("1. upper_direct_substitution")
println("2. lower_direct_substitution")
println("3. gaussian_elimination")
method = parse(Int64, readline())
if method == 1
    c = upper_direct_substitution(A, b)
    println("solution = ", c)
elseif method == 2
    c = lower_direct_substitution(A, b)
    println("solution = ", c)
elseif method == 3
    c = gaussian_elimination(A, b)
    println("solution = ", c)
else
    println("invalid method")
end
