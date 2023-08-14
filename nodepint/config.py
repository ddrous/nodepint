## Configuration file for nodepint package

import jax
import jax.numpy as jnp


__version__ = "0.1.0"       ## Package version

FLOAT64 = False
jax.config.update("jax_enable_x64", FLOAT64)
jnp.set_printoptions(linewidth=jnp.inf)
