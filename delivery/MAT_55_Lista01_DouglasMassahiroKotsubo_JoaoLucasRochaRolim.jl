# =====================================================================
# *********************************************************************
#                    MAT-55 2024 - Lista 01 - Exercício 6 
# *********************************************************************
# =====================================================================

# =====================================================================
#                    Algoritmo de Substituição Direta
# =====================================================================
# ---------------------------------------------------------------------
# Dados de entrada:
# A     matriz nxn triangular inferior, não singular
# b     vetor n
# ---------------------------------------------------------------------
# Saída:
# b     se A é não singular, b é a solução do sistema linear Ax = b

using LinearAlgebra

function sub_direta(A::Matrix{T}, b::Vector{T}; atol::T = 1e-12) where {T <: AbstractFloat}
	@assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
	n = size(A, 1)

	@assert all(all(isapprox.(A[i, (i + 1):n], 0, atol=atol)) for i = 1:n)  "A must be a lower triangular matrix"
	@assert all(!isapprox(A[i, i], 0, atol=atol) for i = 1:n) "A must be a non-singular matrix"

	c = zeros(length(b))

	for i = 1:n
		c[i] = (b[i] - A[i, :]' * c)/A[i, i]
	end

	return c
end

# =====================================================================
#                    Algoritmo de Substituição Inversa
# =====================================================================
# ---------------------------------------------------------------------
# Dados de entrada:
# A     matriz nxn triangular superior, não singular
# b     vetor n
# ---------------------------------------------------------------------
# Saída:
# b     se A é não singular, b é a solução do sistema linear Ax = b
function sub_inversa(A::Matrix{T}, b::Vector{T}; atol::T = 1e-12) where {T <: AbstractFloat}
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

# =====================================================================
#                    Algoritmo de Eliminação Gaussiana
# =====================================================================
# ---------------------------------------------------------------------
# Dados de entrada:
# A     matriz nxn triangular superior, não singular
# b     vetor n
# ---------------------------------------------------------------------
# Saída:
# b     se A é não singular, b é a solução do sistema linear Ax = b

function elim_gauss(A::Matrix{T}, b::Vector{T}; atol::T = 1e-12) where {T <: AbstractFloat}
	@assert size(A, 1) == size(A, 2) == length(b) "A and b dimension mismatch"
	n = size(A, 1)

	@assert !isapprox(det(A), 0; atol=atol) "A must be a non-singular matrix"

	_A = deepcopy(A)
	_b = deepcopy(b)

	for k = 1:(n - 1)
		τ = vcat(zeros(k), _A[(k + 1):n, k] / _A[k, k])
		e = vcat(zeros(k - 1), 1, zeros(n - k))

		M = I - τ * e'

		_A = M * _A
		_b = M * _b
	end

	c = sub_inversa(_A, _b)

	return c
end

# =====================================================================
# =====================================================================
#			PROGRAMA PRINCIPAL
# =====================================================================
# =====================================================================

# Implemente um programa para resolver o sistema linear Ax = b, com 
#opção para o usuário fornecer a matriz A, o vetor b e escolher o método 
#utilizado, dentre as opçõe:
# a: Algoritmo de Substituição Direta.
# b: Algoritmo de Substituição Inversa. 
# C: Eliminação Gaussiana.

# Dados do sistema
#Digite aqui os dados do sistema linear
A = [1 2; 3 4.]
b =  [5; 11.]

# c = sub_direta(A, b)
# c = sub_inversa(A, b)
c = elim_gauss(A, b)

println(c)
