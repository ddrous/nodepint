#%% [markdown]

# \begin{cases}
# \dot{x} = y \\
# \dot{v} = ay - x(b + cx^2)
# \end{cases}

#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Duffing system
def duffing(t, state, a, b, c):
    x, y = state
    dxdt = y
    dydt = a*y - x*(b + c*x**2)
    # dydt = a*y - x*(c*(x**2))
    return [dxdt, dydt]

# Parameters
a, b, c = -1/2., -1, 1/10.

# # Initial conditions
# x0 = 0.0
# v0 = 0.0
# state0 = [x0, v0]

# ## Random initial conditions
# state0 = [np.random.randn(), np.random.randn()]

# Time span
t_span = (0, 40)
# t_eval = np.linspace(t_span[0], t_span[1], 1000)
t_eval = np.arange(t_span[0], t_span[1], 0.01)


# Initialise figures
plt.figure(figsize=(10, 6))

# Solve the ODE
# init_conds = [ [-1, -1], [2,1]]

## Initial conditiosn in a grids: [-2, 2] x [-1, 1] with 16 points in total
# init_conds = np.array(np.meshgrid(np.linspace(-2, 2, 4), np.linspace(-1, 1, 4))).T.reshape(-1, 2)

## 8 initial conds only from one of the attractors
init_conds = np.array([[-0.5, -1], [-0.5, -0.5], [-0.5, 0.5], 
                       [-1.5, 1], 
                    #    [-0.5, 1], 
                       [-1, -1], [-1, -0.5], [-1, 0.5], [-1, 1], 
                       [-2, -1], [-2, -0.5], [-2, 0.5], [-2, 1]])


train_data = []

for state0 in init_conds:
    sol = solve_ivp(duffing, t_span, state0, args=(a, b, c), t_eval=t_eval)

    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(sol.t, sol.y[0], label='Displacement (x)')
    # plt.plot(sol.t, sol.y[1], label='Velocity (v)')
    # plt.xlabel('Time')
    # plt.ylabel('State')
    # plt.title('Duffing System Simulation')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Add to training data
    train_data.append(sol.y.T)

    ## Plot the phase space
    plt.plot(sol.y[0], sol.y[1])
    plt.xlabel('Displacement (x)')
    plt.ylabel('Velocity (y)')
    plt.title('Phase Space')
    plt.grid(True)

plt.show()

## Save the training data
train_data = np.stack(train_data)
np.savez("data/train.npz", X=train_data, t=t_eval)
