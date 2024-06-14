# =====================================================================
# *********************************************************************
#                          MAT-55 2024 - Lista 10
# *********************************************************************
# =====================================================================
# Dupla: Douglas Massahiro Kotsubo e João Lucas Rocha Rolim
#
#
# Para calcular a SVD você pode usar a SVD do pacote LinearAlgebra
# svd(A)

using CSV
using DataFrames
using LinearAlgebra
using Statistics

############################
# Dados do problema:

df = CSV.read("daily-treasury-rates.csv", DataFrame);
vals = Matrix{Float64}(df[:, 2:end]);

############################
# Matriz na forma de desvio de média:

X = diff(vals; dims = 1)
meanX = mean(X; dims = 1)
devX = X .- meanX

############################
# Matriz de covariância:

covX = devX' * devX / (size(X, 1) - 1)

############################
# SVD

U, D, V = svd(covX)

############################
# Análise de componentes principais 

print(D)

# =====================================================================
# 		           Comentários
# =====================================================================
#Digite aqui os seus comentários. Que informção relevante sobre os dados você obteve após aplicar a Análise de Componentes Principais? 

# Observa-se que o primeiro valor singular possui uma ordem de grandeza superior aos outros
# Logo essa é a componente principal da nossa matriz. Isto indica que a maior parte da variação
# da taxa de juros é bem explicada apenas por essa combinação das variáveis aleatórias
