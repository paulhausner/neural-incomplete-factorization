import torch
import numpy as np
import numml as nm

import ilupp

from neuralif.utils import torch_sparse_to_scipy, time_function
from neuralif.models import NeuralIF


class Preconditioner:
    def __init__(self, A, **kwargs):
        self.breakdown = False
        self.nnz = 0
        self.time = 0
        self.n = kwargs.get("n", 0)
    
    def timed_setup(self, A, **kwargs):
        start = time_function()
        self.setup(A, **kwargs)
        stop = time_function()
        self.time = stop - start
    
    def get_inverse(self):
        ones = torch.ones(self.n)
        offset = torch.zeros(1).to(torch.int64)
        
        I = torch.sparse.spdiags(ones, offset, (self.n, self.n))
        I = I.to(torch.float64)
        
        return I
    
    def get_p_matrix(self):
        return self.get_inverse()
    
    def check_breakdown(self, P):
        if np.isnan(np.min(P)):
            self.breakdown = True
    
    def __call__(self, x):
        return x


class JacobiPreconditioner(Preconditioner):
    def __init__(self, A, **kwargs):
        super().__init__(A, **kwargs)
        self.timed_setup(A)
        
        self.nnz = A.shape[0]
    
    def get_p_matrix(self):
        # we need to reinvert the matrix
        diag = 1 / self.P.values()
        offset = torch.zeros(1).to(torch.int64)
        
        Pinv = torch.sparse.spdiags(diag, offset, (self.n, self.n))
        Pinv = Pinv.to(torch.float64)
        
        return Pinv
        
    def get_inverse(self):
        return self.P
    
    def setup(self, A):
        # We choose L = 1/D = diag(1/a11, 1/a22, ..., 1/ann)
        # data = 1 / torch.Tensor(torch.sqrt(A.diagonal()))
        data = 1 / torch.Tensor(A.diagonal())
        indices = torch.vstack((torch.arange(A.shape[0]), torch.arange(A.shape[0])))
        
        p = torch.sparse_coo_tensor(indices, data, size=A.shape)
        self.P = p.to(torch.float64).to_sparse_csr()
    
    def __call__(self, x):
        return self.P@x


class ICholPreconditioner(Preconditioner):
    def __init__(self, A, **kwargs):
        super().__init__(A, **kwargs)
        
        self.timed_setup(A, **kwargs)
        self.nnz = self.L.nnz
        
    def setup(self, A, **kwargs):
        
        fill_in = kwargs.get("fill_in", 0.0)
        threshold = kwargs.get("threshold", 0.0)
        
        if fill_in == 0.0 and threshold == 0.0:
            icholprec = ilupp.ichol0(A.astype(np.float64).tocsr())
        else:
            icholprec = ilupp.icholt(A.astype(np.float64).tocsr(),
                                     add_fill_in=fill_in,
                                     threshold=threshold)
        
        # icholprec = icholprec.astype(np.float32)
        self.check_breakdown(icholprec)
        
        # convert to nummel sparse format
        self.L = nm.sparse.SparseCSRTensor(icholprec)
        self.U = nm.sparse.SparseCSRTensor(icholprec.T)
    
    def get_p_matrix(self):
        return self.L@self.U
      
    def __call__(self, x):
        return fb_solve(self.L, self.U, x)


class ILUPreconditioner(Preconditioner):
    def __init__(self, A, **kwargs):
        super().__init__(A, **kwargs)
        self.timed_setup(A, **kwargs)
        
        # don't count the diagonal twice in the process...
        self.nnz = self.L.nnz + self.U.nnz - A.shape[0]
    
    def get_inverse(self):
        L_inv = torch.inverse(self.L.to_dense())
        U_inv = torch.inverse(self.U.to_dense())
        
        return U_inv@L_inv
    
    def get_p_matrix(self):
        return self.L@self.U
    
    def setup(self, A, **kwargs):
        # compute ILU preconditioner using ilupp
        B = ilupp.ILU0Preconditioner(A.astype(np.float64).tocsr())
        
        L, U = B.factors()
        
        # check breakdowns
        self.check_breakdown(L)
        self.check_breakdown(U)
        
        # convert to nummel sparse format
        self.L = nm.sparse.SparseCSRTensor(L)
        self.U = nm.sparse.SparseCSRTensor(U)

    def __call__(self, x):
        return fb_solve(self.L, self.U, x)


class LearnedPreconditioner(Preconditioner):
    def __init__(self, data, model, **kwargs):
        super().__init__(data, **kwargs)
        
        self.model = model
        self.spd = isinstance(model, NeuralIF)
        
        self.timed_setup(data, **kwargs)
        
        if self.spd:
            self.nnz = self.L.nnz
        else:      
            self.nnz = self.L.nnz + self.U.nnz - data.x.shape[0]
        
    def setup(self, data, **kwargs):
        L, U, _ = self.model(data)
        
        self.L = L.to("cpu").to(torch.float64)
        self.U = U.to("cpu").to(torch.float64)
    
    def get_inverse(self):
        L_inv = torch.inverse(self.L.to_dense())
        U_inv = torch.inverse(self.U.to_dense())
        
        return U_inv@L_inv
    
    def get_p_matrix(self):
        return self.L@self.U
    
    def __call__(self, x):
        return fb_solve(self.L, self.U, x, unit_upper=not self.spd)


def fb_solve(L, U, r, unit_lower=False, unit_upper=False):
    y = L.solve_triangular(upper=False, unit=unit_lower, b=r)
    z = U.solve_triangular(upper=True, unit=unit_upper, b=y)
    return z


def fb_solve_joint(LU, r):
    # Note: solve triangular ignores the values in lower/upper triangle
    y = LU.solve_triangular(upper=False, unit=False, b=r)
    z = LU.solve_triangular(upper=True, unit=False, b=y)
    return z


# generate preconditioner
def get_preconditioner(data, name, **kwargs):
    
    if name == "baseline" or name == "direct":
        return Preconditioner(None, n=data.x.shape[0] ,**kwargs)
    
    elif name == "learned":
        return LearnedPreconditioner(data, **kwargs)
    
    # convert to sparse matrix
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(),
                                    dtype=torch.float64, requires_grad=False)
    A_s = torch_sparse_to_scipy(A)
    
    if name == "ic" or name == "ichol":
        return ICholPreconditioner(A_s, **kwargs)
    
    elif name == "ilu":
        return ILUPreconditioner(A_s, **kwargs)
    
    elif name == "jacobi":
        return JacobiPreconditioner(A_s, n=data.x.shape[0], **kwargs)
        
    else:
        raise NotImplementedError(f"Preconditioner {name} not implemented!")
