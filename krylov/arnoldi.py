import torch


def arnoldi(M, A, r0, m, tol=1e-12):
    """
    This function computes an orthonormal basis
    
    V_m = {v_1,...,v_{m+1}} 
    
    of K_{m+1}(A, r^{(0)}) = span{r^{(0)}, Ar^{(0)}, ..., A^{m}r^{(0)}}.
    
    Input parameters:
    -----------------
      A: array_like
          An (n x n) array.
      
      b: array_like
          Initial vector of length n
      
      m: int
          One less than the dimension of the Krylov subspace. Must be > 0.
      
      r0: array_like 
          Initial residual (length n)
      
      tol: 
          Tolerance for convergence

    Output:
    -------
      Q: numpy.array 
          n x (m + 1) array, the columns are an orthonormal basis of the Krylov subspace.
      
      H: numpy.array
          An (m + 1) x m array. It is the matrix A on basis Q. It is upper Hessenberg.
    """
    
    # Check inputs
    n = A.shape[0]
    d = r0.dtype
    
    # assert A.shape == (n, n) and b.shape == (n,) and r0.shape == (n,), "Matrix and vector dimensions don not match"
    # assert isinstance(m, int) and m >= 0, "m must be a positive integer"
    
    m = min(m, n)
    
    # Initialize matrices
    V = torch.zeros((n, m + 1), dtype=d)
    H = torch.zeros((m + 1, m), dtype=d)
    
    # Normalize input vector and use for Krylov vector
    beta = torch.linalg.norm(r0)
    V[:, 0] = r0 / beta

    for k in range(1, m + 1):
        # Generate a new candidate vector
        w = M(A @ V[:, k - 1]) # Note that here is different from arnoldi_one_iter as we iter over k from 1 to m. 
                               # In arnoldi_one_iter we have k as inputo to the function and we have V[:, k - 1] as k starts at 0.
        
        # Orthogonalization
        for j in range(k):
            H[j, k - 1] = V[:, j] @ w
            w -= H[j, k - 1] * V[:, j]
        
        H[k, k - 1] = torch.linalg.norm(w)

        # Check convergence
        if H[k, k - 1] <= tol:
            return V, H
        
        # Normalize and store the new basis vector
        V[:, k] = w / H[k, k - 1]
    
    return V, H


def arnoldi_step(M, A, V, k, left=True, tol=1e-12):
    
    n = A.shape[0]  # Dimension of the matrix
    d = A.dtype  # Data type of the matrix
    
    # Initialize k + 2 nonzero elements of H along column k
    h_k = torch.zeros(k + 2, dtype=d)

    # Calculate the new vector in the Krylov subspace
    if left:
        v_new = M(A @ V[:, k])
    else:
        v_new = A @ M(V[:, k])
    # Calculate the first k elements of the kth Hessenberg column
    for j in range(k + 1):
        h_k[j] = v_new @ V[:, j]
        v_new -= h_k[j] * V[:, j]

    # Add the k+1 element
    h_k[k + 1] = torch.linalg.norm(v_new)
    
    # Early termination with exact solution
    if h_k[k + 1] <= tol:
        return h_k, None
    
    # Find the new orthogonal vector in the basis of the Krylov subspace
    v_new /= h_k[k + 1]

    return h_k, v_new
