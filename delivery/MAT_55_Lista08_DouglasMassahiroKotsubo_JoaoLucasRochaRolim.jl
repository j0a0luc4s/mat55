using LinearAlgebra
using BlockArrays

function gradiente_conjugado(A::Matrix{T}, b::Vector{T}, x0::Vector{T}; atol::Float64 = 1e-8, kmax::Int64 = 10000) where{T <: AbstractFloat}
    # verificações básicas
    @assert size(A,1) == size(A,2) == size(b) == size(x0) "Matrizes compatíveis"
    n = size(A,1)

    # matriz não singular
    @assert !isapprox(det(A), 0; atol) "Matriz singular"

    # matriz spd
    for i = 1:n
        @assert !isapprox(det(A[1:i,1:i]), 0 ;atol) "Matriz spd"
    end

    count = 0
    prev_x = deepcopy(x0)
    x = deepcopy(x0)
    d = vec(zeros(n))
    r = deepcopy(d)
    aux = vec(zeros(n))

    for i = 1:n
        d[i] = b[i] - dot(A[i, :], x)
        r[i] = d[i]
    end

    while count < kmax
        for i = 1:n
            aux[i] = dot(A[i, :], d)
        end
        α = dot(r,d)/(dot(d,aux))
        
        x = x + α*d
        for i = 1:n
            r[i] = b[i] - dot(A[i,:], x)
        end

        β = -dot(r,aux)/(dot(d,aux))
        d = r + β*d
        count += 1

        if norm(x-prev_x)/norm(x) < atol
            break
        end
        prev_x = x
    end

    println("O numero de iterações foi: ", count)
    return x;
end