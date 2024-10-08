import torch


def stopping_criterion(A, rk, b):
    return torch.inner(rk, rk) / torch.inner(b, b)


def conjugate_gradient(A, b, x0=None, x_true=None, rtol=1e-8, max_iter=100_000):
    x_hat = x0 if x0 is not None else torch.zeros_like(b)
    r = b - A@x_hat # residual
    p = r.clone() # search direction
    
    # Errors is a tuple of (error, residual)
    error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
    res = stopping_criterion(A, r, b)
    errors = [(torch.inner(error_i, A@error_i), res)]
    
    for _ in range(max_iter):
        if res < rtol:
            break
        
        Ap = A@p
        r_norm = torch.inner(r, r)
        
        a = r_norm / torch.inner(Ap, p) # step length
        x_hat = x_hat + a * p
        r = r - a * Ap
        p = r + (torch.inner(r, r) / r_norm) * p
        
        error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
        res = stopping_criterion(A, r, b)
        errors.append((torch.inner(error_i, A@error_i), res))
        
    return errors, x_hat


def preconditioned_conjugate_gradient(A, b, M=None, x0=None, x_true=None, rtol=1e-8, max_iter=100_000):
    # prec should be a function solving the linear equation system Mz=r one way or another
    # M is the preconditioner approximation of A^-1 or split approximation of MM^T=A
    # Saad, 2003 Algorithm 9.1
    
    if M is None:
        M = lambda x: x
    
    x_hat = x0 if x0 is not None else torch.zeros_like(b)
    
    rk = b - A@x_hat
    zk = M(rk)
    pk = zk.clone()
    
    # Errors is a tuple of (error, residual)
    error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
    res = stopping_criterion(A, zk, b)
    errors = [(torch.inner(error_i, A@error_i), res)]
    
    for _ in range(max_iter):
        if res < rtol:
            break
        
        # precomputations
        Ap = A@pk
        rz = torch.inner(rk, zk)
        
        a = rz / torch.inner(Ap, pk) # step length
        x_hat = x_hat + a * pk
        rk = rk - a * Ap
        zk = M(rk)
        beta = torch.inner(rk, zk) / rz
        pk = zk + beta * pk
        
        error_i = (x_hat - x_true) if x_true is not None else torch.zeros_like(b, requires_grad=False)
        res = stopping_criterion(A, rk, b)
        errors.append((torch.inner(error_i, A@error_i), res))
        
    return errors, x_hat
