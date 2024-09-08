import argparse
import os
import datetime

import numpy as np
import scipy
import scipy.sparse
import torch
import json

from krylov.cg import conjugate_gradient, preconditioned_conjugate_gradient
from krylov.gmres import gmres
from krylov.preconditioner import get_preconditioner

from neuralif.models import NeuralIF, NeuralPCG, PreCondNet, LearnedLU
from neuralif.utils import torch_sparse_to_scipy, time_function
from neuralif.logger import TestResults

from apps.data import matrix_to_graph_sparse, get_dataloader


@torch.inference_mode()
def test(model, test_loader, device, folder, save_results=False, dataset="random", solver="cg"):
    
    if save_results:
        os.makedirs(folder, exist_ok=False)

    print()
    print(f"Test:\t{len(test_loader.dataset)} samples")
    print(f"Solver:\t{solver} solver")
    print()
    
    # Two modes: either test baselines or the learned preconditioner
    if model is None:
        methods = ["baseline", "jacobi", "ilu"]
    else:
        assert solver in ["cg", "gmres"], "Data-driven method only works with CG or GMRES"
        methods = ["learned"]
    
    # using direct solver
    if solver == "direct":
        methods = ["direct"]
    
    for method in methods:
        print(f"Testing {method} preconditioner")
        
        test_results = TestResults(method, dataset, folder,
                                   model_name= f"\n{model.__class__.__name__}" if method == "learned" else "",
                                   target=1e-6,
                                   solver=solver)
        
        for sample, data in enumerate(test_loader):
            plot = save_results and sample == (len(test_loader.dataset) - 1)
            
            # Getting the preconditioners
            start = time_function()
            
            data = data.to(device)
            prec = get_preconditioner(data, method, model=model)
            
            # Get properties...
            p_time = prec.time
            breakdown = prec.breakdown
            nnzL = prec.nnz
            
            stop = time_function()
            
            A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(),
                                        dtype=torch.float64,
                                        requires_grad=False).to("cpu").to_sparse_csr()
            
            b = data.x[:, 0].squeeze().to("cpu").to(torch.float64)
            b_norm = torch.linalg.norm(b)
            
            # we assume that b is unit norm wlog
            b = b / b_norm
            solution = data.s.to("cpu").to(torch.float64).squeeze() / b_norm if hasattr(data, "s") else None
            
            # if we run single sample, create accurate solution (for testing purposes...)
            # if solution is None and len(test_loader.dataset) == 1 and save_results:
            #    print("Generate dummy solution...")
            #    solution = torch.rand_like(b)
            #    b = A@solution
            
            overhead = (stop - start) - (p_time)
            
            # RUN CONJUGATE GRADIENT
            start_solver = time_function()
            
            solver_settings = {
                "max_iter": 10_000,
                "x0": None
            }
            
            if breakdown:
                res = []
            
            elif solver == "direct":
                
                # convert to sparse matrix (scipy)
                A_ = torch.sparse_coo_tensor(data.edge_index, data.edge_attr.squeeze(),
                                             dtype=torch.float64, requires_grad=False)
                
                # scipy sparse...
                A_s = torch_sparse_to_scipy(A_).tocsr()
                
                # override start time
                start_solver = time_function()
                
                dense = False
                
                if dense:
                    _ = scipy.linalg.solve(A_.to_dense().numpy(), b.numpy(), assume_a='pos')
                else:
                    _ = scipy.sparse.linalg.spsolve(A_s, b.numpy())
                
                # dummy values...
                res = [(torch.Tensor([0]), torch.Tensor([0]))] * 2
            
            elif solver == "cg" and method == "baseline":
                # no preconditioner required when using baseline method
                res, _ = conjugate_gradient(A, b, x_true=solution,
                                            rtol=test_results.target, **solver_settings)
            
            elif solver == "cg":
                res, _ = preconditioned_conjugate_gradient(A, b, M=prec, x_true=solution,
                                                           rtol=test_results.target, **solver_settings)
                
            elif solver == "gmres":
                
                res, _ = gmres(A, b, M=prec, x_true=solution,
                               **solver_settings, plot=plot,
                               atol=test_results.target,
                               left=False)
            
            stop_solver = time_function()
            solver_time = (stop_solver - start_solver)
            
            # LOGGING
            test_results.log_solve(A.shape[0], solver_time, len(res) - 1,
                                   np.array([r[0].item() for r in res]),
                                   np.array([r[1].item() for r in res]),
                                   p_time, overhead)
            
            # ANALYSIS of the preconditioner and its effects!
            nnzA = A._nnz()
            
            test_results.log(nnzA, nnzL, plot=plot)
            
            svd = False
            if svd:
                # compute largest and smallest singular value
                Pinv = prec.get_inverse()
                APinv = A.to_dense() @ Pinv
                
                # compute the singular values of the preconditioned matrix
                S = torch.linalg.svdvals(APinv)
                
                # print the smallest and largest singular value
                test_results.log_eigenval_dist(S, plot=plot)
                
                # compute the loss of the preconditioner
                p = prec.get_p_matrix()
                loss1 = torch.linalg.norm(p.to_dense() - A.to_dense(), ord="fro")
                
                a_inv = torch.linalg.inv(A.to_dense())
                loss2 = torch.linalg.norm(p.to_dense()@a_inv - torch.eye(a_inv.shape[0]), ord="fro")
                
                test_results.log_loss(loss1, loss2, plot=False)
                
                print(f"Smallest singular value: {S[-1]} | Largest singular value: {S[0]} | Condition number: {S[0] / S[-1]}")
                print(f"Loss Lmax: {loss1}\tLoss Lmin: {loss2}")
                print()
                
        if save_results:
            test_results.save_results()
        
        test_results.print_summary()


def load_checkpoint(model, args, device):
    # load the saved weights of the model and the hyper-parameters
    checkpoint = args.checkpoint
    
    if checkpoint == "latest":
        # list all the directories in the results folder
        d = os.listdir("./results/")
        d.sort()
        
        config = None
        
        # find the latest checkpoint
        for i in range(len(d)):
            if os.path.isdir("./results/" + d[-i-1]):
                dir_contents = os.listdir("./results/" + d[-i-1])
                
                # looking for a directory with both config and model weights
                if "config.json" in dir_contents and "final_model.pt" in dir_contents:
                    # load the config.json file
                    with open("./results/" + d[-i-1] + "/config.json") as f:
                        config = json.load(f)
                        
                        if config["model"] != args.model:
                            config = None
                            continue
                        
                        if "best_model.pt" in dir_contents:
                            checkpoint = "./results/" + d[-i-1] + "/best_model.pt"
                            break
                        else:
                            checkpoint = "./results/" + d[-i-1] + "/final_model.pt"
                            break
        if config is None:
            print("Checkpoint not found...")
        
        # neuralif has optional drop tolerance...
        if args.model == "neuralif":
            config["drop_tol"] = args.drop_tol
        
        # intialize model and hyper-parameters
        model = model(**config)
        print(f"load checkpoint: {checkpoint}")
        
        model.load_state_dict(torch.load(checkpoint, weights_only=False, map_location=torch.device(device)))
    
    elif checkpoint is not None:
        with open(checkpoint + "/config.json") as f:
            config = json.load(f)
        
        if args.model == "neuralif":
            config["drop_tol"] = args.drop_tol
        
        model = model(**config)
        print(f"load checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint + f"/{args.weights}.pt",
                                            map_location=torch.device(model.device)))
    
    else:
        model = model(**{"global_features": 0, "latent_size": 8, "augment_nodes": False,
                            "message_passing_steps": 3, "skip_connections": True, "activation": "relu",
                            "aggregate": None, "decode_nodes": False})
        
        print("No checkpoint provided, using random weights")
    
    return model


def warmup(model, device):
    # set testing parameters
    model.to(device)
    model.eval()
    
    # run model warmup
    test_size = 1_000
    matrix = scipy.sparse.coo_matrix((np.ones(test_size), (np.arange(test_size), np.arange(test_size))))
    data = matrix_to_graph_sparse(matrix, torch.ones(test_size))
    data.to(device)
    _ = model(data)
    
    print("Model warmup done...")


# argument is the model to load and the dataset to evaluate on
def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=int, required=False)
    
    # select data driven model to run
    parser.add_argument("--model", type=str, required=False, default="none")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--weights", type=str, required=False, default="model")
    parser.add_argument("--drop_tol", type=float, default=0)
    
    parser.add_argument("--solver", type=str, default="cg")
    
    # select dataset and subset
    parser.add_argument("--dataset", type=str, required=False, default="random")
    parser.add_argument("--subset", type=str, required=False, default="test")
    parser.add_argument("--n", type=int, required=False, default=0)
    parser.add_argument("--samples", type=int, required=False, default=None)
    
    # select if to save
    parser.add_argument("--save", action='store_true', default=False)
    
    return parser.parse_args()


def main():
    args = argparser()
    
    if args.device is not None:
        test_device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        test_device = "cpu"
        
    if args.name is not None:
        folder = "results/" + args.name
    else:
        folder = folder = "results/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    print()
    print(f"Using device: {test_device}")
    # torch.set_num_threads(1)
    
    # Load the model
    if args.model == "nif" or args.model == "neuralif":
        print("Use model: NeuralIF")
        model = NeuralIF
    
    elif args.model == "lu" or args.model == "learnedlu":
        print("Use model: LU")
        model = LearnedLU
        
        assert args.solver == "gmres", "LU only supports GMRES solver"
    
    elif args.model == "neural_pcg" or args.model == "neuralpcg":
        print("Use model: NeuralPCG")
        model = NeuralPCG
    
    elif args.model == "precondnet":
        print("Use model: precondnet")
        model = PreCondNet
    
    elif args.model == "none":
        print("Running non-data-driven baselines")
        model = None
    
    else:
        raise NotImplementedError(f"Model {args.model} not available.")
    
    if model is not None:
        model = load_checkpoint(model, args, test_device)
        warmup(model, test_device)
    
    spd = args.solver == "cg" or args.solver == "direct"
    testdata_loader = get_dataloader(args.dataset, n=args.n, batch_size=1, mode=args.subset,
                                     size=args.samples, spd=spd, graph=True)
    
    # Evaluate the model
    test(model, testdata_loader, test_device, folder,
         save_results=args.save, dataset=args.dataset, solver=args.solver)


if __name__ == "__main__":
    main()
