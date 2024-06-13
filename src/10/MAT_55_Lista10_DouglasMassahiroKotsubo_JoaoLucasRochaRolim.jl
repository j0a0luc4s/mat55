# =====================================================================
# *********************************************************************
#                          MAT-55 2024 - Lista 10
# *********************************************************************
# =====================================================================
# Dupla: Douglas Massahiro Kotsubo e João Lucas Rocha Rolim
#
#
#Para calcular a SVD você pode usar a SVD do pacote LinearAlgebra
#svd(A)
using LinearAlgebra, CSV, DataFrames

#Dados do problema:
X = CSV.read("daily-treasury-rates.csv", DataFrame; header = false);
Z = Matrix(X);
l = size(X,1); c = size(X,2);
Y = Z[2:l,2:c];
Y = parse.(Float64, Y);

############################
#Matriz na forma de desvio de média:
A = zeros(l-1,c-1);
for i = 1:l-2
    A[i,:] = Y[i,:] - Y[i+1,:];
end

############################
#Matriz de covariância:
K = A'*A;
K = A/(l-1);

############################
#SVD
U, S, V = svd(K);

# =====================================================================
# 		           Comentários
# =====================================================================
#Digite aqui os seus comentários. Que informção relevante sobre os dados você obteve após aplicar a Análise de Componentes Principais? 
#
#Observa-se que os valores singulares possuem ordens de grandeza semelhantes o que indica
#que as informações contidas na matriz original são todas relevantes.
#
