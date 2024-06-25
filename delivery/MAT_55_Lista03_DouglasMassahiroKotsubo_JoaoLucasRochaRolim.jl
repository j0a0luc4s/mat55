using LinearAlgebra

function upper_direct_substitution(
    A::AbstractMatrix{Tp},
    b::AbstractVector{Tp};
    atol::Tp=sqrt(eps(Tp))
) where {Tp<:AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    @assert all(all(isapprox.(A[i, 1:i - 1], 0, atol=atol)) for i = 1:n) "A must be an upper triangular matrix"
    @assert all(!isapprox(A[i, i], 0, atol=atol) for i = 1:n) "A must be a non-singular matrix"

    c = zeros(Tp, n)

    for i = n:-1:1
        c[i] = (b[i] - A[i, :]' * c) / A[i, i]
    end

    return c
end

function lower_direct_substitution(
    A::AbstractMatrix{Tp},
    b::AbstractVector{Tp};
    atol::Tp=sqrt(eps(Tp))
) where {Tp<:AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    @assert all(all(isapprox.(A[i, i + 1:n], 0, atol=atol)) for i = 1:n) "A must be a lower triangular matrix"
    @assert all(!isapprox(A[i, i], 0, atol=atol) for i = 1:n) "A must be a non-singular matrix"

    c = zeros(Tp, n)

    for i = 1:n
        c[i] = (b[i] - A[i, :]' * c) / A[i, i]
    end

    return c
end

function gaussian_elimination(
    A::AbstractMatrix{Tp},
    b::AbstractVector{Tp};
    atol::Tp=sqrt(eps(Tp))
) where {Tp<:AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
    n = size(A, 1)

    U = deepcopy(A)
    b = deepcopy(b)

    for k = 1:n - 1
        _, i = findmax(abs.(U[k:n, k]))
        i = i + k - 1
        @assert !isapprox(U[i, k], 0; atol=atol) "A must be a non-singular matrix"

        U[i, :], U[k, :] = U[k, :], U[i, :]
        b[i], b[k] = b[k], b[i]

        τ = vcat(zeros(Tp, k), U[k + 1:n, k] / U[k, k])
        e = vcat(zeros(Tp, k - 1), 1, zeros(Tp, n - k))

        M = I - τ * e'

        U = M * U
        b = M * b
    end

    c = upper_direct_substitution(U, b; atol=atol)

    return c
end

function lu_decomposition_nopivot(
    A::AbstractMatrix{Tp};
    atol::Tp=sqrt(eps(Tp))
) where {Tp<:AbstractFloat}
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

function lu_decomposition_pivot(
    A::AbstractMatrix{Tp};
    atol::Tp=sqrt(eps(Tp))
) where {Tp<:AbstractFloat}
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
println("Select a method (type 1-5): ")
println("1. upper_direct_substitution")
println("2. lower_direct_substitution")
println("3. gaussian_elimination")
println("4. lu_decomposition_nopivot")
println("5. lu_decomposition_pivot")
println("6. cholesky_decomposition")
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
elseif method == 4
    L, U = lu_decomposition_nopivot(A)
    c = upper_direct_substitution(U, lower_direct_substitution(L, b))
    println("L = ")
    for i in 1:n
        println(L[i, :])
    end
    println("U = ")
    for i in 1:n
        println(U[i, :])
    end
    println("solution = ", c)
elseif method == 5
    L, U, P = lu_decomposition_pivot(A)
    c = upper_direct_substitution(U, lower_direct_substitution(L, P * b))
    println("L = ")
    for i in 1:n
        println(L[i, :])
    end
    println("U = ")
    for i in 1:n
        println(U[i, :])
    end
    println("P = ")
    for i in 1:n
        println(P[i, :])
    end
    println("solution = ", c)
elseif method == 6
    G = cholesky_decomposition(A)
    c = upper_direct_substitution(G', lower_direct_substitution(G, b))
    println("G = ")
    for i in 1:n
        println(G[i, :])
    end
    println("solution = ", c)
else
    println("invalid method")
end
