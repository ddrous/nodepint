# NodePinT

Parallel in Time exploration of Neural ODEs.

## Features
- Feature mapping to various problems (ODEs)
- Time-parallelisation (or layer parallelisation if compared to ResNets)
- For optimal control (OC) while training a neural ODE, we propose several combinations:
    - DP for training, and DAL for OC
    - DP both for training and OC (a bit like PINN)
    - Same as above, but DAL for training


## Getting started
`pip install nodepint`


## ToDos
- Massive parallelisation by combining time with data
- Stochastic ODEs and diffusion models time parallelisation



## Flowchart

![Flowchart](docs/imgs/flowchart.png)



## Dependencies
- JAX
- Equinox
- Julia bindings to Python for SPH (all papers should benchmark NODEs on physical data upon us realeasing the SPHPC dataset)
