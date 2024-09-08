import os
import datetime
import argparse
import pprint
import torch
import torch_geometric
import time

from apps.data import get_dataloader, graph_to_matrix

from neuralif.utils import count_parameters, save_dict_to_file, condition_number, eigenval_distribution, gershgorin_norm
from neuralif.logger import TrainResults, TestResults
from neuralif.loss import loss
from neuralif.models import NeuralPCG, NeuralIF, PreCondNet, LearnedLU

from krylov.cg import preconditioned_conjugate_gradient
from krylov.gmres import gmres
from krylov.preconditioner import LearnedPreconditioner


@torch.no_grad()
def validate(model, validation_loader, solve=False, solver="cg", **kwargs):
    model.eval()
                
    acc_loss = 0.0
    num_loss = 0
    acc_solver_iters = 0.0
    
    for i, data in enumerate(validation_loader):
        data = data.to(device)
        
        # construct problem data
        A, b = graph_to_matrix(data)
        
        # run conjugate gradient method
        # this requires the learned preconditioner to be reasonably good!
        if solve:
            # run CG on CPU
            with torch.inference_mode():
                preconditioner = LearnedPreconditioner(data, model)
            
            A = A.to("cpu").to(torch.float64)
            b = b.to("cpu").to(torch.float64)
            x_init = None
            
            solver_start = time.time()
            
            if solver == "cg":
                l, x_hat = preconditioned_conjugate_gradient(A.to("cpu"), b.to("cpu"), M=preconditioner,
                                                             x0=x_init, rtol=1e-6, max_iter=1_000)
            elif solver == "gmres":
                l, x_hat = gmres(A, b, M=preconditioner, x0=x_init, atol=1e-6, max_iter=1_000, left=False)
            else:
                raise NotImplementedError("Solver not implemented choose between CG and GMRES!")
            
            solver_stop = time.time()
            
            # Measure preconditioning performance
            solver_time = (solver_stop - solver_start)
            acc_solver_iters += len(l) - 1
        
        else:
            output, _, _ = model(data)
            
            # Here, we compute the loss using the full forbenius norm (no estimator)
            # l = frobenius_loss(output, A)
            
            l = loss(data, output, config="frobenius")
            
            acc_loss += l.item()
            num_loss += 1
    
    if solve:
        # print(f"Smallest eigenvalue: {dist[0]}")
        print(f"Validation\t iterations:\t{acc_solver_iters / len(validation_loader):.2f}")
        return acc_solver_iters / len(validation_loader)
        
    else:
        print(f"Validation loss:\t{acc_loss / num_loss:.2f}")
        return acc_loss / len(validation_loader)


def main(config):
    if config["save"]:
        os.makedirs(folder, exist_ok=True)
        save_dict_to_file(config, os.path.join(folder, "config.json"))
    
    # global seed-ish
    torch_geometric.seed_everything(config["seed"])
    
    # args for the model
    model_args = {k: config[k] for k in ["latent_size", "message_passing_steps", "skip_connections",
                                         "augment_nodes", "global_features", "decode_nodes",
                                         "normalize_diag", "activation", "aggregate", "graph_norm",
                                         "two_hop", "edge_features", "normalize"]
                  if k in config}
    
    # run the GMRES algorithm instead of CG (?)
    gmres = False
    
    # Create model
    if config["model"] == "neuralpcg":
        model = NeuralPCG(**model_args)
    
    elif config["model"] == "nif" or config["model"] == "neuralif" or config["model"] == "inf":
        model = NeuralIF(**model_args)
        
    elif config["model"] == "precondnet":
        model = PreCondNet(**model_args)
        
    elif config["model"] == "lu" or config["model"] == "learnedlu":
        gmres = True
        model = LearnedLU(**model_args)
        
    else:
        raise NotImplementedError
    
    model.to(device)
    
    print(f"Number params in model: {count_parameters(model)}")
    print()
    
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    
    # Setup datasets
    train_loader = get_dataloader(config["dataset"], config["n"], config["batch_size"],
                                  spd=not gmres, mode="train")
    
    validation_loader = get_dataloader(config["dataset"], config["n"], 1, spd=(not gmres), mode="val")
    
    best_val = float("inf")
    logger = TrainResults(folder)
    
    # todo: compile the model
    # compiled_model = torch.compile(model, mode="reduce-overhead")
    # model = torch_geometric.compile(model, mode="reduce-overhead")
    
    total_it = 0
    
    # Train loop
    for epoch in range(config["num_epochs"]):
        running_loss = 0.0
        grad_norm = 0.0
        
        start_epoch = time.perf_counter()
        
        for it, data in enumerate(train_loader):
            # increase iteration count
            total_it += 1
            
            # enable training mode
            model.train()
            
            start = time.perf_counter()
            data = data.to(device)
            
            output, reg, _ = model(data)
            l = loss(output, data, c=reg, config=config["loss"])
            
            #  if reg:
            #    l = l + config["regularizer"] * reg
            
            l.backward()
            running_loss += l.item()
            
            # track the gradient norm
            if "gradient_clipping" in config and config["gradient_clipping"]:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
            
            else:
                total_norm = 0.0
                
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
            
                grad_norm = total_norm ** 0.5 / config["batch_size"]
            
            # update network parameters
            optimizer.step()
            optimizer.zero_grad()
        
            logger.log(l.item(), grad_norm, time.perf_counter() - start)
            
            # Do validation after 100 updates (to support big datasets)
            # convergence is expected to be pretty fast...
            if (total_it + 1) % 1000 == 0:
                
                # start with cg-checks after 5 iterations
                val_its = validate(model, validation_loader, solve=True,
                                    solver="gmres" if gmres else "cg")
                    
                # use scheduler
                # if config["scheduler"]:
                #    scheduler.step(val_loss)
                
                logger.log_val(None, val_its)
                
                # val_perf = val_cgits if val_cgits > 0 else val_loss
                val_perf = val_its
                
                if val_perf < best_val:
                    if config["save"]:
                        torch.save(model.state_dict(), f"{folder}/best_model.pt")
                    best_val = val_perf
        
        epoch_time = time.perf_counter() - start_epoch
        
        # save model every epoch for analysis...
        if config["save"]:
            torch.save(model.state_dict(), f"{folder}/model_epoch{epoch+1}.pt")
        
        print(f"Epoch {epoch+1} \t loss: {1/len(train_loader) * running_loss} \t time: {epoch_time}")
    
    # save fully trained model
    if config["save"]:
        logger.save_results()
        torch.save(model.to(torch.float).state_dict(), f"{folder}/final_model.pt")
    
    # Test the model
    # wandb.run.summary["validation_chol"] = best_val
    print()
    print("Best validation loss:", best_val)


def argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=int, required=False)
    parser.add_argument("--save", action='store_true')
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="random")
    parser.add_argument("--loss", type=str, required=False)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    
    parser.add_argument("--regularizer", type=float, default=0)
    parser.add_argument("--scheduler", action='store_true', default=False)
    
    # parser.add_argument("--training_samples", type=int, default=100)
    # parser.add_argument("--optimizer", type=str, default="adam")
    # parser.add_argument("--learning_rate", type=float, default=1e-3)
    # parser.add_argument("--weight_decay", type=float, default=0)
    
    # Model parameters
    parser.add_argument("--model", type=str, default="neuralif")
    
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--message_passing_steps", type=int, default=3)
    parser.add_argument("--decode_nodes", action='store_true', default=False)
    parser.add_argument("--normalize_diag", action='store_true', default=False)
    parser.add_argument("--aggregate", nargs="*", type=str)
    parser.add_argument("--activation", type=str, default="relu")
    
    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--encoder_layer_norm", action='store_true', default=False)
    # parser.add_argument("--mp_layer_norm", action='store_true', default=False)
    # parser.add_argument("--k", type=int, default=1)
    
    # NIF parameters
    parser.add_argument("--skip_connections", action='store_true', default=True)
    parser.add_argument("--augment_nodes", action='store_true')
    parser.add_argument("--global_features", type=int, default=0)
    parser.add_argument("--edge_features", type=int, default=1)
    parser.add_argument("--graph_norm", action='store_true')
    parser.add_argument("--two_hop", action='store_true')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    
    if args.device is None:
        device = "cpu"
        print("Warning!! Using cpu only training")
        print("If you have a GPU use that with the command --device {id}")
        print()
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    if args.name is not None:
        folder = "results/" + args.name
    else:
        folder = folder = "results/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    print(f"Using device: {device}")
    print("Using config: ")
    pprint.pprint(vars(args))
    print()
    
    # run experiments
    main(vars(args))
