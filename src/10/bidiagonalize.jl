function bidiagonalize(
    A::Matrix{Tp};
    atol::Tp = 1e-6
) where {Tp <: AbstractFloat}
    m = size(A, 1)
    n = size(A, 2)

    @assert m >= n "A must be an m x n matrix, where m >= n"

    B = deepcopy(A)

    UWUYt = zeros(m, m)
    VWVYt = zeros(n, n)

    for j = 1:n
        u, β = house(B[j:m, j]; atol=atol)
        B[j:m, j:n] = (I - β*u*u')*B[j:m, j:n]

        u = vcat(zeros(j - 1), u)
        UWUYt = UWUYt + (I - UWUYt)*β*u*u'

        if j + 2 <= n
            v, β = house(B'[j + 1:n, j]; atol=atol)
            B[j:m, j + 1:n] = B[j:m, j + 1:n]*(I - β*v*v')

            v = vcat(zeros(j), v)
            VWVYt = VWVYt + (I - VWVYt)*β*v*v'
        end
    end

    U = I - UWUYt
    V = I - VWVYt

    return U, B, V
end
