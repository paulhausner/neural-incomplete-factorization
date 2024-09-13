# Neural incomplete factorization

This repository contains the code for learning incomplete factorization preconditioners directly from data by [Paul Häusner](https://paulhausner.github.io), Aleix Nieto Juscafresa, [Ozan Öktem](https://www.kth.se/profile/ozan), and [Jens Sjölund](https://jsjol.github.io/).

## Installation

In order to run the training and testing, you need to install the following python dependencies:

- pytorch
- pytorch-geometric
- scipy
- networkx

For validation and testing the following packages are required:

- matplotlib
- [numml](https://github.com/nicknytko/numml) (for efficient forward-backward substitution)
- [ilupp](https://github.com/c-f-h/ilupp) (for baseline incomplete factorization preconditioners)

## Implementation

The repository consists of several parts. In the `krylov` folder implementations for the conjugate gradient method and GRMES method are provided. Further, several preconditioner (Jacobi, ILU, IC) are implemented.

The `neuralif` module contains the code for the learned preconditioner. The model.py file contains the different models that can be utilizes, loss.py implements several different loss functions.

A synthetic dataset is provided in the folder `apps`.

## References

If our code helps your research or work, please consider citing our paper. The following are BibTeX references:

```
@article{hausner2023neural,
  title={Neural incomplete factorization: learning preconditioners for the conjugate gradient method},
  author={H{\"a}usner, Paul and {\"O}ktem, Ozan and Sj{\"o}lund, Jens},
  journal={arXiv preprint arXiv:2305.16368},
  year={2023}
}

@article{hausner2024learning,
  title={Learning incomplete factorization preconditioners for {GMRES}},
  author={H{\"a}usner, Paul and Nieto Juscafresa, Aleix and Sj{\"o}lund, Jens},
  journal={arXiv preprint arXiv:2409.08262},
  year={2024}
}
```

Please feel free to reach out if you have any questions or comments.

Contact: Paul Häusner, paul.hausner@it.uu.se
