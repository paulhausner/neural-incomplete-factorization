import torch

from krylov.arnoldi import arnoldi, arnoldi_step


def back_substitution(A, b):
    """
    Solve a linear system using back substitution.
    
    !!! Onyl for testing purposes, use torch.linalg.solve_triangular instead !!!
    
    Args:
    ----------
        A: Coefficient matrix (must be upper triangular).
        b: Column vector of constants.
    
    Returns:
    --------
        list: Solution vector.
        
    Raises: ValueError: If the matrix A is not square or if its dimensions are incompatible with the vector b.
    """
    
    n = len(b)
    
    # Check if A is a square matrix
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Matrix A must be square.")
    
    # Check if dimensions of A and b are compatible
    if len(A) != len(b):
        raise ValueError("Dimensions of A and b are incompatible.")
    
    x = torch.zeros(n, dtype=b.dtype)
    
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - torch.sum(A[i, i+1:] * x[i+1:])) / A[i, i]
    
    return x


def gmres(A, b, M=None, left=True, x0=None, x_true=None, atol=1e-8, rtol=None, max_iter=100_000, restart=None, plot=False):
    """
    Restarted Generalized Minimal RESidual method for solving linear systems.
    
    Parameters:
    -----------
    A : Coefficient matrix of the linear system.
        
    b : Right-hand side vector of the linear system.
        
    M : Preconditioner operator needs to allow M(x) to be computed for any vector x.
        
    x0 : Initial guess for the solution.
        
    k_max : Maximum number of iterations. When None  we set k_max to the dimension of A.
        
    restart : Number of iterations before restart. If None, the method will not restart.
        
    rtol, atol : Tolerance for convergence.
    
    plot : If True, plot the convergence of the method (makes algorithm slower).
    
    Returns:
    --------
    errors : Residual and error at each iteration.
    
    pk : Norm of the residual vector.
    """
    
    n = A.shape[0]
    
    if max_iter is None or max_iter > n:
        max_iter = n
    
    if x0 is None:
        x0 = torch.zeros(n, dtype=b.dtype)
        
    if M is None:
        # identity preconditioner
        M = lambda x: x
    
    if left:
        r0 = M(b - A @ x0)
    else:
        r0 = b - A @ x0
    
    p0 = torch.linalg.norm(r0)
    
    pk = p0.clone()
    beta = p0.clone()
    
    def compute_solution(R, Q, V, beta, x0):
        # yk = back_substitution(R[:-1, :], beta*Q[0][:-1])
        yk = torch.linalg.solve_triangular(R[:-1, :], beta * Q[0][:-1].reshape(-1, 1), upper=True)
        yk = yk.reshape(-1)
        if left:
            xk = x0 + V[:, :-1]@yk # Compute the new approximation x0 + V_{k}y
        else:
            xk = x0 + M(V[:, :-1]@yk)
        
        return xk
    
    error_i = (x0 - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
    errors = [(torch.linalg.norm(error_i), p0)]
    
    # Intialize the Arnoldi algorithm
    V_ = r0.clone().reshape(-1, 1) / beta
    H_ = torch.zeros((n+1, 1), dtype=b.dtype)
    
    k = 0
    for i in range(max_iter):
        
        # ARNOLDI ALGORITHM for krylov basis
        # Arnoldi algorithm to generate V_{k+1} and H_{K+1, K}
        # V, H = arnoldi(M, A, r0, k+1)
        
        h_new, v_new = arnoldi_step(M, A, V_, k, left=left)
        H_ = torch.cat((H_, torch.zeros((n + 1, 1))), axis=1)
        H_[:k+2, -1] = h_new
        
        if v_new is None:
            # found exact solution (this does not happen in practice)
            # ? not sure if we need to recompute the QR decomposition
            Q, R = torch.linalg.qr(H_[:k+2, 1:], mode='complete') # system of size m
            V_ = torch.cat((V_, torch.zeros(n, 1)), axis=1)
            errors.append((0, 0)) # logging reasons
            break
        
        V_ = torch.cat((V_, v_new.reshape(-1, 1)), axis=1)
        
        # QR DECOMPOSITION
        # TODO: can be achieved with rotation matrices afaik
        Q, R = torch.linalg.qr(H_[:k+2, 1:], mode='complete') # system of size m
        pk = torch.abs(beta*Q[0, k+1]) # Compute norm of residual vector
        
        k += 1
        
        # LOGGING
        if plot:
            xk = compute_solution(R, Q, V_, beta, x0)
            error_i = (xk - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
            errors.append((torch.norm(error_i), pk))
        else:
            errors.append((errors[-1][0], pk))
        
        # STOPPING CRITERIA
        if atol is not None and pk < atol:
            break
        if rtol is not None and pk < rtol*p0:
            break
        
        # RESTART (don't restart if we are in the last iteration)
        elif restart is not None and k == restart and i < max_iter - 1:
            
            # Compute current solution
            x0 = compute_solution(R, Q, V_, beta, x0)
            
            # Reset iterates
            r0 = M(b - A @ x0)
            p0 = torch.linalg.norm(r0)
            beta = p0.clone()
            pk = p0.clone()
            k = 0
            
            # Reset Arnoldi algorithm
            V_ = r0.clone().reshape(-1, 1) / beta
            H_ = torch.zeros((n+1, 1), dtype=b.dtype)
        
    xk = compute_solution(R, Q, V_, beta, x0)
    return errors, xk
