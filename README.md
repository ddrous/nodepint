# NodePinT

Parallel in Time exploration of Neural ODEs.

## Features
- Feature mapping to various problems (ODEs)
- Time-parallelisation (or layer parallelisation if compared to ResNets)
    - Time is split into chunks, and the ODE is solved in parallel for each chunk (1st level)
    - Since the PinT problem is projected on a lower dimensional space, this allows for further parallelism (2nd level):
        - If we always project on a randomly sampled 1D space, we can also solve the ODEs in parallel and average the NN weights without having to re(jit)compile
        - If we project on an increasingly bigger randomly sampled spaces, we can still do it, but we need to re(jit)compile the PinT processes
        - If we construct our space in a deterministic way (via sensitivity analysis wrt the latest added vector), then we can't parallelise since it will be sequential.
- For optimal control (OC) while training a neural ODE, we propose several combinations:
    - DP for training, and DAL for OC
    - DP both for training and OC (a bit like PINN)
    - Same as above, but DAL for training
- Functional programming (FP), by treating each Neural ODE problem as a state (a class) to be passed each functions
    - Always pass classes to the functions, even for visualisation
    - Make the functions as pure as possible: no side effects, even for random number generators
    - Bundle all your side effects (when the function modifies something outside of its scope, e.g. the screen, the disk, memory, etc.) into the main section
    - Note: Create classes from NamedTuple to carry states. Then use jax.tree_map to update the states: https://jax.readthedocs.io/en/latest/jax-101/07-state.html


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
