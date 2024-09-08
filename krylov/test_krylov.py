import unittest

import torch
import numpy as np

from krylov.arnoldi import arnoldi, arnoldi_step
from krylov.cg import conjugate_gradient, preconditioned_conjugate_gradient
from krylov.gmres import gmres, back_substitution


class TestKrylov(unittest.TestCase):
    
    def test_arnoldi(self):
        n = 100
        m = 4
        
        M = lambda x: x
        A = torch.rand(n, n, dtype=torch.float64)
        b = torch.rand(n, dtype=torch.float64)
        x0 = torch.zeros(n, dtype=torch.float64)
        
        r0 = M(b - A @ x0)

        V, H = arnoldi(M, A, r0, m)
        
        assert torch.linalg.cond(V) - 1 < 1e-5, f"Condition number of V is {torch.linalg.cond(V)}"
        assert torch.linalg.norm(V.T @ V - torch.eye(m + 1, dtype=torch.float64)) < 1e-5, "V is not orthogonal"
        
        for i in range(1, m + 2):
            cond_number = torch.linalg.cond(V[:,:i])
            assert cond_number - 1 < 1e-5, f"Condition number of V[:,:{i}] is {cond_number}"
        
        # A@V[:,:-1] = V@H
        torch.testing.assert_close(A@V[:,:-1], V@H, rtol=1e-5, atol=1e-5)

    def test_arnoldi_step(self):
        n = 100
        m = 5
        
        M = lambda x: x
        A = torch.rand(n, n, dtype=torch.float64)
        b = torch.rand(n, dtype=torch.float64)
        x0 = torch.zeros(n, dtype=torch.float64)
        r0 = M(b - A @ x0)
        
        # classical approach
        Q, H = arnoldi(M, A, r0, m)

        # initalize the V matrix
        V = torch.zeros((n, 1), dtype=torch.float64)
        beta = np.linalg.norm(r0)
        
        V[:, 0] = r0 / beta
        R = torch.zeros((m + 1, 1), dtype=torch.float64)
        
        for i in range(m):
            h, v = arnoldi_step(M, A, V, i)
            
            V = torch.cat((V, torch.zeros((n, 1), dtype=torch.float64)), axis=1)
            V[:, i + 1] = v
            
            R = torch.cat((R, torch.zeros((m + 1, 1), dtype=torch.float64)), axis=1)
            R[:(i + 2), i] = h
            
        torch.testing.assert_close(Q, V, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(H, R[:, :-1], rtol=1e-5, atol=1e-5)
            
    def test_gmres(self):
        A, b = discretise_poisson(10)
        e, x_hat = gmres(A, b)
        
        x_direct = torch.linalg.inv(A.to_dense()) @ b
        torch.testing.assert_close(x_hat, x_direct, rtol=1e-5, atol=1e-5)
    
    def test_gmres_restart(self):
        A, b = discretise_poisson(10)
        _, x_hat = gmres(A, b, restart=10)
        _, x_hat2 = gmres(A, b)
        x_direct = torch.linalg.inv(A.to_dense()) @ b
        
        torch.testing.assert_close(x_hat, x_direct, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(x_hat, x_hat2, rtol=1e-5, atol=1e-5)
    
    def test_gmres_preconditioner(self):
        A, b = discretise_poisson(10)
        
        M = torch.diag(torch.rand(100, dtype=torch.float64)).to_sparse_coo()
        precond = lambda x: M@x
        
        e, x_hat = gmres(A, b, M=precond)
        x_direct = torch.linalg.inv(A.to_dense()) @ b
        
        torch.testing.assert_close(x_hat, x_direct, rtol=1e-5, atol=1e-5)
    
    def test_gmres_right(self):
        A, b = discretise_poisson(10)
        M = torch.diag(torch.rand(100, dtype=torch.float64)).to_sparse_coo()
        precond = lambda x: M@x
        
        e, x_hat = gmres(A, b, M=precond, left=False)
        
        x_direct = torch.linalg.inv(A.to_dense()) @ b
        torch.testing.assert_close(x_hat, x_direct, rtol=1e-5, atol=1e-5)
    
    def test_gmres_identity(self):
        A = torch.eye(100, dtype=torch.float64)
        b = torch.rand(100, dtype=torch.float64)
        
        e, x_hat = gmres(A, b)
        
        assert len(e) == 2, f"GMRES converged in {len(e) - 1} iterations"
        torch.testing.assert_close(x_hat, b, rtol=1e-5, atol=1e-5)
    
    def test_gmres_vs_cg(self):
        A, b = discretise_poisson(10)
        
        _, x_cg = conjugate_gradient(A, b)
        _, x_gmres = gmres(A, b)
        
        torch.testing.assert_close(x_cg, x_gmres, rtol=1e-5, atol=1e-5)
    
    def test_conjugate_gradient(self):
        A = torch.rand(100, 100, dtype=torch.float64)
        A = A @ A.T + 0.1 * torch.eye(100, dtype=torch.float64)
        x = torch.rand(100, dtype=torch.float64)
        
        # obtain rhs and normalize
        b = A @ x
        b = b / torch.linalg.norm(b)
        
        _, x_hat = conjugate_gradient(A, b, x, rtol=1e-8, max_iter=100_000)
        
        # check that the solution has a small residual
        res_norm = torch.linalg.norm(A @ x_hat - b) / torch.linalg.norm(b)
        
        assert res_norm < 1e-3, f"Residual norm is {res_norm}"
    
    def test_cg_preconditioner(self):
        A = torch.rand(100, 100, dtype=torch.float64)
        A = A @ A.T + 0.1 * torch.eye(100, dtype=torch.float64)
        
        M = torch.rand(100, 100, dtype=torch.float64)
        M = M @ M.T + 0.1 * torch.eye(100, dtype=torch.float64)
        precond = lambda x: M@x
        
        b = torch.rand(100, dtype=torch.float64)
        
        _, x_hat_1 = conjugate_gradient(A, b, x_true=None, rtol=1e-15)
        _, x_hat_2 = preconditioned_conjugate_gradient(A, b, M=precond, x_true=None, rtol=1e-15)
        
        torch.testing.assert_close(x_hat_1, x_hat_2, rtol=1e-6, atol=1e-6)
    
    def test_backsubsitution(self):
        
        # create data
        A = torch.rand(100, 100, dtype=torch.float64)
        U = torch.triu(A)
        b = torch.rand((100, 1), dtype=torch.float64)
        
        # solve the system using back substitution
        x = back_substitution(U, b)
        x_true = torch.linalg.solve_triangular(U, b, upper=True)
        
        torch.testing.assert_close(x, x_true.squeeze(), rtol=1e-5, atol=1e-5)
    

def discretise_poisson(N):
    """Generate the matrix and rhs associated with the discrete Poisson operator."""
    
    nelements = 5 * N**2 - 16 * N + 16
    
    row_ind = np.zeros(nelements, dtype=np.float64)
    col_ind = np.zeros(nelements, dtype=np.float64)
    data = np.zeros(nelements, dtype=np.float64)
    
    f = np.zeros(N * N, dtype=np.float64)
    
    count = 0
    for j in range(N):
        for i in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                row_ind[count] = col_ind[count] = j * N + i
                data[count] =  1
                f[j * N + i] = 0
                count += 1
                
            else:
                row_ind[count : count + 5] = j * N + i
                col_ind[count] = j * N + i
                col_ind[count + 1] = j * N + i + 1
                col_ind[count + 2] = j * N + i - 1
                col_ind[count + 3] = (j + 1) * N + i
                col_ind[count + 4] = (j - 1) * N + i
                                
                data[count] = 4 * (N - 1)**2
                data[count + 1 : count + 5] = - (N - 1)**2
                f[j * N + i] = 1
                
                count += 5
    
    # create the sparse pytorch matrix
    idx = np.vstack((row_ind, col_ind))
    A = torch.sparse_coo_tensor(idx, data, (N**2, N**2)).coalesce()
    b = torch.tensor(f)
    
    return A, b


if __name__ == '__main__':
    unittest.main()
