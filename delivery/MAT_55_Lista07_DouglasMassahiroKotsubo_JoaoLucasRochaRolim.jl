# =====================================================================
# *********************************************************************
#                    MAT-55 2024 - Lista 7
# *********************************************************************
# =====================================================================
# Dupla:
# Douglas Massahiro Kotsubo
# João Lucas Rocha Rolim
# =====================================================================
#                        Método de Jacobi
# =====================================================================
# Dados de entrada:
# a     matriz, nxn não singular
# b	vetor, n
# x	vetor, n, aproximação inicial
# tol	escalar, tolerância para o critério de parada
# kmax	escalar, número máximo de iterações permitido
#
# Saída:
# x     aproximação para a solução do sistema Ax=b

using LinearAlgebra
using BlockArrays

function jacobi(A::Matrix{T}, b::Vector{T}, x0::Vector{T}; atol::T = 1e-8, k::Int64 = 10000) where{T <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) == length(x0) "A and b dimension mismatch"
    n = size(A,1)

    @assert !isapprox(det(A), 0; atol=atol) "A must be a non-singular matrix"
    @assert !any(isapprox.(diag(A), 0; atol=atol)) "Null pivot"

    prev_x = deepcopy(x0)
    x = deepcopy(x0)
    count = 0
    while count < k
        for i = 1:n
            x[i] = (b[i] - dot(A[i, :], prev_x) + A[i, i]*prev_x[i])/A[i, i]
        end
        count += 1;
        if norm(x - prev_x)/norm(prev_x) < atol
            break;
        end
        prev_x = deepcopy(x)
    end

    return x, count
end 

# =====================================================================
#                        Método de Gauss-Seidel
# =====================================================================
# Dados de entrada:
# a     matriz, nxn não singular
# b	vetor, n
# x	vetor, n, aproximação inicial
# tol	escalar, tolerância para o critério de parada
# kmax	escalar, número máximo de iterações permitido
#
# Saída:
# x     aproximação para a solução do sistema Ax=b

function gauss_seidel(A::Matrix{T}, b::Vector{T}, x0::Vector{T}; atol::T = 1e-8, k::Int64 = 10000) where{T <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) == length(x0) "A and b dimension mismatch"
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

# =====================================================================
#                        Método SOR
# =====================================================================
# Dados de entrada:
# a     matriz, nxn não singular
# b	vetor, n
# x	vetor, n, aproximação inicial
# w	escalar, parâmetro do método SOR
# tol	escalar, tolerância para o critério de parada
# kmax	escalar, número máximo de iterações permitido
#
# Saída:
# x     aproximação para a solução do sistema Ax=b

function sor(A::Matrix{T}, b::Vector{T}, x0::Vector{T}, w::T; atol::T = 1e-8, k::Int64 = 10000) where{T <: AbstractFloat}
    @assert size(A, 1) == size(A, 2) == length(b) == length(x0) "A and b dimension mismatch"
    n = size(A,1)

    @assert !isapprox(det(A), 0; atol=atol) "A must be a non-singular matrix"
    @assert !any(isapprox.(diag(A), 0; atol=atol)) "Null pivot"
    @assert 0 < w < 2 "w ∉ (0, 2)"
    
    prev_x = deepcopy(x0)
    x = deepcopy(x0)
    count = 0;
    while count < k
        for i = 1:n
            x[i] = (b[i] - dot(A[i, :], x) + A[i, i]*x[i])/A[i,i]
            x[i] = prev_x[i] + w*(x[i] - prev_x[i])
        end
        count = count + 1;
        if norm(x - prev_x)/norm(prev_x) < atol
            break;
        end
        prev_x = deepcopy(x)
    end

    return x, count
end

function tridiagm(dminus1::Vector{T}, d0::Vector{T}, d1::Vector{T}) where {T <: AbstractFloat}
    @assert length(dminus1) + 1 == length(d0) == length(d1) + 1 "dminus1, d0 and d1 dimension mismatch"
    n = length(d0)

    M = zeros(Float64, n, n)

    M[diagind(M, -1)] = dminus1
    M[diagind(M, 0)] = d0
    M[diagind(M, 1)] = d1

    return M;
end

for m = Int64[5, 10, 20, 40]

    # =====================================================================
    # 			Dados do problema
    # =====================================================================

    T = tridiagm(-1*ones(m-2), 4*ones(m-1), -1*ones(m-2))
    Id = Matrix{Float64}(I, m-1, m-1)
    z = zeros(Float64, m-1, m-1)

    A = BlockArray{Float64}(undef_blocks, (m-1)*ones(Int64, m-1), (m-1)*ones(Int64, m-1))
    for i = 1:m-1
        for j = 1:m-1
            if i == j
                setblock!(A, T, i, j)
            elseif abs(i - j) == 1
                setblock!(A, -Id, i, j)
            else
                setblock!(A, z, i, j)
            end
        end
    end
    A = Matrix{Float64}(A)

    b = zeros((m-1)^2)

    for i = 1:m-1
        b[i] = (i/m - 1)*sin(i/m)
    end
    b[m-1] += 1/m

    for i = 2:m-2
        b[(m-1)*i] = i/m
    end

    for i =1:m-1
        b[(m-1)^2-(m-1)+i] = (i/m)*(2-i/m)
    end
    b[(m-1)^2] += (m-1)/m

    x0 = zeros(Float64, (m-1)^2)

    # =====================================================================
    # 		         Resultados obtidos
    # =====================================================================

    println("Jabobi method for m = " * string(m))
    @time x, count = jacobi(A, b, x0)
    println("iterations: ", count)

    println("Gauss Seidel method for m = " * string(m))
    @time x, count = gauss_seidel(A, b, x0)
    println("iterations: ", count)

    println("SOR method for m = " * string(m))
    for w = 1.0 : 0.1 : 1.9
        println("w = " * string(w))
        @time x, count = sor(A, b, x0, w)
        println("iterations: ", count)
    end
end

# =====================================================================
# 		           Comentários
# =====================================================================

# O experimento confirma que o método de Jacobi é o mais lento e o que
# exige mais iterações pra convergência

# O experimento confirma que o método SOR é tão bom ou melhor que o de 
# Gauss Seidel no conjunto explorado, sendo no mínimo tão bom quanto

# Para valores grandes de m no método SOR temos que w mais próximo de 2
# gera resultados melhores no conjunto explorado
