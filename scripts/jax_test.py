import numpy as np
import jax.numpy as jnp
import jax
import time


## Some commands to quickly try stuff
# sbatch scripts/jade.sh  # Submit the script to JADE
# sacct -u rrn27-wwp02    # Monitor my jobs


print("Jax devices are:", jax.devices())
print("Jax version:", jax.__version__)


def f(x):  # function we're benchmarking (works in both NumPy & JAX)
  return x.T @ (x - x.mean(axis=0))

start = time.time()
x_np = np.ones((100, 100), dtype=np.float32)  # same as JAX default dtype
f(x_np)  # measure NumPy runtime

# x_jax = jax.device_put(x_np)  # measure JAX device transfer time
# f_jit = jax.jit(f)
# f_jit(x_jax).block_until_ready()  # measure JAX compilation time
# f_jit(x_jax).block_until_ready()  # measure JAX runtime

end = time.time()
print("Total time:", end-start)
