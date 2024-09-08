import os

from dataclasses import dataclass, field
from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from neuralif.utils import kA_bound


@dataclass
class TestResults:
    method: str
    dataset: str
    folder: str
    
    # for learned distinguish between different models
    model_name: str = ""
    
    # general parameters
    seed: int = 0
    target: float = 1e-8
    solver: str = "cg"
        
    # store the results of the test evaluation
    n: List[int] = field(default_factory=list)
    # cond_pa: List[float] = field(default_factory=list)
    nnz_a: List[float] = field(default_factory=list)
    nnz_p: List[float] = field(default_factory=list)
    p_times: List[float] = field(default_factory=list)
    overhead: List[float] = field(default_factory=list)
    
    # store results from solver (cg or gmres) run
    solver_time: List[float] = field(default_factory=list)
    solver_iterations: List[float] = field(default_factory=list)
    solver_error: List[float] = field(default_factory=list)
    solver_residual: List[float] = field(default_factory=list)
    
    # more advanved loggings (not always set)
    distribution: List[torch.Tensor] = field(default_factory=list)
    loss1: List[float] = field(default_factory=list)
    loss2: List[float] = field(default_factory=list)
    
    def log(self, nnz_a, nnz_p, plot=False):
        
        self.nnz_a.append(nnz_a)
        self.nnz_p.append(nnz_p)
        
        if plot:
            self.plot_convergence()
    
    def log_solve(self, n, solver_time, solver_iterations, solver_error, solver_residual, p_time, overhead):
        self.n.append(n)
        self.solver_time.append(solver_time)
        self.solver_iterations.append(solver_iterations)
        self.solver_error.append(solver_error)
        self.solver_residual.append(solver_residual)
        self.p_times.append(p_time)
        self.overhead.append(overhead)
    
    def log_eigenval_dist(self, dist, plot=False):
        # eigenvalue of singular value dist :)
        
        self.distribution.append(dist.numpy())
        
        if plot:
            self.plot_eigvals(dist)
    
    def log_loss(self, loss1, loss2, plot=False):
        self.loss1.append(loss1)
        self.loss2.append(loss2)
        
        if plot:
            self.plot_loss()
    
    def plot_convergence(self):
        
        # check convergence speed etc.
        error_0 = self.solver_error[-1][0]
        # errors = [fun(r[0]) for r in res]
        # residuals = [fun(r[1]) for r in res]
        
        if self.solver == "cg" and False:
            bounds = [error_0 * kA_bound(self.cond_pa[-1], k) for k in range(len(self.solver_residual[-1]))]
        else:
            bounds = None
        
        plt.plot(self.solver_error[-1], label="error ($|| x_i - x_* ||$)")
        plt.plot(self.solver_residual[-1], label="residual ($||r ||_2$)")
        
        if bounds is not None:
            plt.plot(bounds, "--", label="k(A)-bound")
        
        plt.plot([self.target for _ in self.solver_residual[-1]], ":")
        
        plt.grid(alpha=0.3)
        
        plt.yscale("log")
        plt.title(f"Convergence: {self.method} in {len(self.solver_residual[-1]) - 1} iterations")
        plt.xlabel("iteration")
        plt.ylabel("log10")
        plt.legend()
        
        sample = len(self.solver_time)
        plt.savefig(f"{self.folder}/convergence_{self.solver}_{self.method}_{sample}.pdf")
        plt.close()
    
    def plot_eigvals(self, dist, name=""):
        
        c = torch.max(dist) / torch.min(dist)
        
        # plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 14
        
        plt.grid(alpha=0.3)
        
        bins=20
        # bins=[0, 0.01, 0.1, 0.2,0.3,0.5,1,1.5,2,3,4,5]
        plt.hist(dist.tolist(), density=True, bins=bins, alpha=0.7)
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(dist.tolist())
        plt.plot(kde_xs, kde.pdf(kde_xs), "--", alpha=0.7)
        
        # plt.xscale("log")
        # plt.xlim(right=2)
        
        plt.title(f"$\kappa(A)=${c.item():.2e}")
        plt.ylabel("Frequency")
        plt.xlabel("$\lambda$")
        plt.savefig(f"{self.folder}/eigenvalues_{self.method}_{name}.png")
        plt.close()
    
    def plot_loss(self):
        i = len(self.solver_time) - 1
            
        # plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 14
        
        fig, axs = plt.subplots(1, 3, figsize=plt.figaspect(1/3))
        # fig.suptitle(f"{self.method.upper()} Error: {self.loss[-1]:.2f}" + m)
        
        im1 = axs[0].imshow(torch.abs(self.A), interpolation='none', cmap='Blues')
        im1.set_clim(0, 1)
        axs[0].set_title("A")
        im2 = axs[1].imshow(torch.abs(self.L), interpolation='none', cmap='Blues')
        im2.set_clim(0, 1)
        axs[1].set_title("L")
        
        res = torch.abs(self.L@self.L.T - self.A)
        
        # show at points where A is non-zero
        # res = torch.where(torch.abs(self.A) > 0, res, torch.zeros_like(res))
        
        im3 = axs[2].imshow(res, interpolation='none', cmap='Reds')
        im3.set_clim(0, 1)
        axs[2].set_title("L@L.T - A")
        
        # add colorbat
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.4, hspace=0.1)
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        fig.colorbar(im3,cax=cb_ax)
        
        # share y-axis
        for ax in fig.get_axes():
            ax.label_outer()
                        
        # save as file
        plt.savefig(f"{self.folder}/chol_factorization_{self.method}_{i}.png")
        plt.close()
    
    def print_summary(self):
        for key, value in self.get_summary_dict().items():
            print(f"{key}:\t{value}")
        print()
        
    def get_total_p_time(self):
        # needs to return an array with the total time required for the preconditioner
        # p_time + inv_time + overhead
        return [p + o for p, o in zip(self.p_times, self.overhead)]
        
    def get_summary_dict(self):
        # check where ch did not break down
        valid_samples = np.asarray(self.solver_iterations) > 0
        
        data = {
            f"time_{self.method}": np.mean(self.p_times, where=valid_samples),
            f"overhead_{self.method}": np.mean(self.overhead, where=valid_samples),
            f"{self.solver}_time_{self.method}": np.mean(self.solver_time, where=valid_samples),
            f"{self.solver}_iterations_{self.method}": np.mean(self.solver_iterations, where=valid_samples),
            f"total_time_{self.method}": np.mean(list(map(lambda x: x[0] + x[1], zip(self.get_total_p_time(), self.solver_time))), where=valid_samples),
            f"time-per-iter": np.sum(self.solver_time, where=valid_samples) / np.sum(self.solver_iterations, where=valid_samples),
            f"nnz_a_{self.method}": np.mean(self.nnz_a),
            f"nnz_p_{self.method}": np.mean(self.nnz_p),
        }
        
        # add information about failure runs...
        if np.sum(valid_samples) < len(self.solver_iterations):
            data = {**data, **{f"success_rate_{self.method}": np.sum(valid_samples) / len(self.solver_iterations)}}
            
        return data
    
    def save_results(self):
        fn = f"{self.folder}/test_{self.method}.npz"
        
        np.savez(fn, n=self.n,
                 p_time=self.p_times,
                 overhead_time=self.overhead,
                 nnz_a=self.nnz_a,
                 nnz_p=self.nnz_p,
                 solver=self.solver,
                 solver_time=self.solver_time,
                 solver_iterations=self.solver_iterations,
                 solver_error=np.asarray(self.solver_error, dtype="object"),
                 solver_residual=np.asarray(self.solver_residual, dtype="object"),
                 eig_distribution=np.asarray(self.distribution, dtype="object"),
                 loss1=self.loss1,
                 loss2=self.loss2)
    
    
@dataclass
class TrainResults:
    folder: str
    
    # training
    loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    
    # validation
    log_freq: int = 100
    val_loss: List[float] = field(default_factory=list)
    val_its: List[float] = field(default_factory=list)
    
    def log(self, loss, grad_norm, time):
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        self.times.append(time)
    
    def log_val(self, val_loss, val_its):
        self.val_loss.append(val_loss)
        self.val_its.append(val_its)
        
    def save_results(self):
        fn = f"{self.folder}/training.npz"
        np.savez(fn, loss=self.loss, grad_norm=self.grad_norm,
                 val_loss=self.val_loss, val_cond=self.val_its)


def create_folder(folder=None):
    if folder is None:
        folder = f"./results/{os.path.basename(__file__).split('.')[0]}"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    return folder
