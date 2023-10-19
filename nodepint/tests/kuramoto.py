#%%

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the Kuramoto model as a system of first-order ODEs
def kuramoto_model(theta, t, K):
    N = len(theta)
    dtheta = np.zeros(N)
    for i in range(N):
        sum_term = np.sum(np.sin(theta - theta[i]))
        dtheta[i] = 1.0 + (K / N) * sum_term  # Natural frequencies are set to 1.0
    return dtheta

# Set up parameters for synchronized behavior
N = 100  # Number of oscillators
K = 2.0  # Strong coupling strength
omega = np.ones(N)  # All oscillators have the same natural frequency
theta0 = np.random.uniform(0, 2 * np.pi, N)  # Initial phases

# Time parameters
t_max = 10.0  # Maximum simulation time
t = np.linspace(0, t_max, 1000)  # Time points for integration

# Solve the ODE system using odeint
theta = odeint(kuramoto_model, theta0, t, args=(K,))

# Calculate the Kuramoto order parameter over time
order_parameter = np.abs(np.mean(np.exp(1j * theta), axis=1))

# Visualize the order parameter and oscillator phases
plt.figure(figsize=(10, 4))

plt.subplot(2, 1, 1)
plt.plot(t, order_parameter, label='Order Parameter (R)')
plt.xlabel('Time')
plt.ylabel('Order Parameter (R)')
plt.title('Kuramoto Model Order Parameter')
plt.legend()

plt.subplot(2, 1, 2)
for i in range(N):
    plt.plot(t, np.unwrap(theta)[:, i])
plt.xlabel('Time')
plt.ylabel('Oscillator Phase')
plt.title('Kuramoto Model Oscillator Phases')
plt.legend()

plt.tight_layout()
plt.show()


# %%
