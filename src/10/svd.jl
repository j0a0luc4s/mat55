function golub_kahan_svd_step(
    B::Matrix{Tp};
    atol::Tp = 1e-6
) where {Tp <: AbstractFloat}
    m = size(B, 1)
    n = size(B, 2)

    @assert m >= n >= 2 "B must be an m x n matrix, where m >= n >= 2"

    @assert !any(isapprox.(diag(B, 0), 0.0; atol=atol))
            !any(isapprox.(diag(B, 1), 0.0; atol=atol))
            "A must have no zeros on its diagonal or superdiagonal"

    U = diagm(m, m, ones(m))
    Bnew = deepcopy(B)
    V = diagm(n, n, ones(n))

    T22 = Bnew[1:n, n - 1:n]'*Bnew[1:n, n - 1:n]

    μ, _ = findmin(x -> abs(x - T22[2, 2]), eigvals(T22))

    y = Bnew[1:n, 1]'*Bnew[1:n, 1] - μ
    z = Bnew[1:n, 1]'*Bnew[1:n, 2]

    for k = 1:n - 1
        G = givens(y, z; atol=atol)
        Bnew[:, [k, k + 1]] = Bnew[:, [k, k + 1]]*G
        V[:, [k, k + 1]] = V[:, [k, k + 1]]*G
        y = Bnew[k, k]
        z = Bnew[k + 1, k]
        G = givens(y, z; atol=atol)
        Bnew[[k, k + 1], :] = G'*Bnew[[k, k + 1], :]
        U[:, [k, k + 1]] = U[:, [k, k + 1]]*G
        if k < n - 1
            y = Bnew[k, k + 1]
            z = Bnew[k, k + 2]
        end
    end

    return U, Bnew, V
end



function svd(
    A::Matrix{Tp};
    atol::Tp = 1e-6
) where {Tp <: AbstractFloat}
    m = size(A, 1)
    n = size(A, 2)

    @assert m >= n "A must be an m x n matrix, where m >= n"

    U, B, V = bidiagonalize(A; atol=atol)

    q = 0
    p = n

    while true
        while n - q - 1 > 0 && isapprox(B[n - q - 1, n - q], 0.0; atol=atol)
            q += 1
        end
        if n - q - 1 == 0
            break
        end
        p = n - q - 2
        while 0 < p && !isapprox(B[p, p + 1], 0.0; atol=atol)
            p -= 1
        end
        if any(isapprox.(diag(B[p + 1:n - q, p + 1:n - q]), 0.0; atol=atol))
            for i = p + 1:n - q - 1
                if isapprox(B[i, i], 0.0; atol=atol)
                    for j = i + 1:n - q
                        G = givens(B[j, j], B[i, j]; atol=atol)
                        B[[i, j], :] = G'*B[[i, j], :]
                        U[:, [i, j]] = U[:, [i, j]]*G
                    end
                end
            end
            j = n - q
            if isapprox(B[j, j], 0.0; atol=atol)
                for i = j - 1:-1:1
                    G = givens(B[i, i], B[i, j]; atol=atol)
                    B[:, [i, j]] = B[:, [i, j]]*G'
                    V[:, [i, j]] = V[:, [i, j]]*G
                end
            end
        else
            U22, B22, V22 = golub_kahan_svd_step(B[p + 1:n - q, p + 1:n - q]; atol=atol)
            B[p + 1:n - q, p + 1:n - q] = B22;
            U[:, p + 1:n - q] = U[:, p + 1:n - q]*U22
            V[:, p + 1:n - q] = V[:, p + 1:n - q]*V22
        end
    end

    return U, B, V
end
