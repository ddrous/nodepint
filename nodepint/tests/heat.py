#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the 2D heat equation ODE function
def heat_equation(u, t, alpha, dx, dy, Nx, Ny):
    u = u.reshape((Nx, Ny))
    dudt = np.zeros_like(u)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            dudt[i, j] = alpha * ((u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2)
    return dudt.reshape(-1)

# Set up parameters
Nx, Ny = 50, 50  # Number of spatial points in x and y directions
Lx, Ly = 1.0, 1.0  # Spatial domain dimensions
alpha = 0.01  # Thermal diffusivity
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Spatial steps

# Set initial temperature distribution (e.g., a Gaussian)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
u0 = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)

# Flatten the initial condition for odeint
u0_flat = u0.reshape(-1)

# Time parameters
t_max = 0.5  # Maximum simulation time
t = np.linspace(0, t_max, 100)  # Time points for integration

# Solve the 2D heat equation using odeint
u = odeint(heat_equation, u0_flat, t, args=(alpha, dx, dy, Nx, Ny))

# Reshape the solution for visualization
u = u.reshape(-1, Nx, Ny)

# Visualize the temperature distribution over time in a 2D plot
fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(X, Y, u[-1], cmap='coolwarm', levels=50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'2D Heat Equation Simulation at t = {t_max:.2f}')
cbar = plt.colorbar(c)
cbar.set_label('Temperature')
plt.show()


# %%

from matplotlib.animation import FuncAnimation

def update(frame):
    ax.clear()
    c = ax.contourf(X, Y, u[frame], cmap='coolwarm', levels=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'2D Heat Equation Simulation at t = {t[frame]:.2f}')

ani = FuncAnimation(fig, update, frames=len(t), repeat=False)

# Save the animation as a GIF file
ani.save('heat_equation_animation.gif', writer='pillow', fps=10)

plt.show()

# %%

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the 3D heat equation ODE function
def heat_equation(u, t, alpha, dx, dy, dz, Nx, Ny, Nz):
    u = u.reshape((Nx, Ny, Nz))
    dudt = np.zeros_like(u)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                dudt[i, j, k] = alpha * (
                    (u[i+1, j, k] - 2*u[i, j, k] + u[i-1, j, k]) / dx**2 +
                    (u[i, j+1, k] - 2*u[i, j, k] + u[i, j-1, k]) / dy**2 +
                    (u[i, j, k+1] - 2*u[i, j, k] + u[i, j, k-1]) / dz**2
                )
    return dudt.reshape(-1)

# Set up parameters
Nx, Ny, Nz = 30, 30, 30  # Number of spatial points in x, y, and z directions
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Spatial domain dimensions
alpha = 0.01  # Thermal diffusivity
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)  # Spatial steps

# Set initial temperature distribution (e.g., a Gaussian)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z)
u0 = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2) / 0.1)

# Flatten the initial condition for odeint
u0_flat = u0.reshape(-1)

# Time parameters
t_equilibrium = np.array([1.0])  # Array of time positions to reach equilibrium
t_max = 2.0 * t_equilibrium[-1]  # Maximum simulation time, including the equilibrium period
t = np.linspace(0, t_max, 200)  # More frames to ensure equilibrium is reached
fps = 10  # Frame rate for the animation

# Solve the 3D heat equation to reach equilibrium
u_equilibrium = odeint(heat_equation, u0_flat, t_equilibrium, args=(alpha, dx, dy, dz, Nx, Ny, Nz))

# Start the animation from the equilibrium state
u0_flat = u_equilibrium[-1]

# Create an empty 3D scatter plot for the initial frame
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Take a slice of the 3D temperature distribution to create a 2D array for the initial frame
initial_frame = u0[Nx // 2, :, :]

x_slice = X[Nx // 2, :, :]
y_slice = Y[Nx // 2, :, :]
z_slice = Z[Nx // 2, :, :]

scatter = ax.scatter(x_slice, y_slice, z_slice, c=initial_frame, cmap='coolwarm')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'3D Heat Equation Simulation at t = 0.00')
cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)
cbar.set_label('Temperature')

# Initialize the scatter plot with the initial data
scatter_frame = [scatter]

def update(frame):
    # Calculate the temperature distribution for the current frame
    u_frame = odeint(heat_equation, u0_flat, [t[frame]], args=(alpha, dx, dy, dz, Nx, Ny, Nz))
    u_frame = u_frame.reshape((Nx, Ny, Nz))
    
    # Take a slice of the 3D temperature distribution to create a 2D array for the current frame
    current_frame = u_frame[Nx // 2, :, :]
    
    # Update the scatter plot with the new data
    scatter = ax.scatter(x_slice, y_slice, z_slice, c=current_frame, cmap='coolwarm')
    scatter.set_array(current_frame.ravel())
    ax.set_title(f'3D Heat Equation Simulation at t = {t[frame]:.2f}')
    
    # Remove the old scatter plot and add the new one
    scatter_frame[0].remove()
    scatter_frame[0] = scatter

ani = FuncAnimation(fig, update, frames=len(t), repeat=False)

plt.show()




# %%

import numpy as np
from scipy.integrate import odeint
import pyvista as pv

# Set the rendering backend to pyvistaqt
# pv.start_xvfb()
pv.set_jupyter_backend("ipyvtklink")
pv.set_plot_theme("document")


# Define the 3D heat equation ODE function
def heat_equation(u, t, alpha, dx, dy, dz, Nx, Ny, Nz):
    u = u.reshape((Nx, Ny, Nz))
    dudt = np.zeros_like(u)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                dudt[i, j, k] = alpha * (
                    (u[i+1, j, k] - 2*u[i, j, k] + u[i-1, j, k]) / dx**2 +
                    (u[i, j+1, k] - 2*u[i, j, k] + u[i, j-1, k]) / dy**2 +
                    (u[i, j, k+1] - 2*u[i, j, k] + u[i, j, k-1]) / dz**2
                )
    return dudt.reshape(-1)

# Set up parameters
Nx, Ny, Nz = 30, 30, 30  # Number of spatial points in x, y, and z directions
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Spatial domain dimensions
alpha = 0.01  # Thermal diffusivity
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)  # Spatial steps

# Set initial temperature distribution (e.g., a Gaussian)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z)
u0 = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2) / 0.1)

# Flatten the initial condition for odeint
u0_flat = u0.reshape(-1)

# Time parameters
t_equilibrium = np.array([1.0])  # Array of time positions to reach equilibrium
t_max = 2.0 * t_equilibrium[-1]  # Maximum simulation time, including the equilibrium period
t = np.linspace(0, t_max, 200)  # More frames to ensure equilibrium is reached

# Solve the 3D heat equation to reach equilibrium
u_equilibrium = odeint(heat_equation, u0_flat, t_equilibrium, args=(alpha, dx, dy, dz, Nx, Ny, Nz))

# Start the animation from the equilibrium state
u0_flat = u_equilibrium[-1]

# Create a PyVista grid from the temperature distribution
grid = pv.ImageData()
grid.dimensions = [Nx, Ny, Nz]
grid.spacing = [Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)]
grid.origin = [0.0, 0.0, 0.0]
temperature = u0_flat.reshape((Nx, Ny, Nz), order='F')

print(temperature)

# Add the temperature as point data to the grid
grid.point_data["Temperature"] = temperature.ravel(order='F')

# Create a plotter and add the grid with the temperature
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="Temperature", cmap="coolwarm")

# Show the plot
plotter.show(auto_close=False)

# %%

import pyvista as pv
import numpy as np
pv.start_xvfb()

# backends = ["ipyvtklink", "panel", "ipygany", "static", "pythreejs", "client", "server", "trame", "none"]
# backends = ["ipyvtklink", "static", "client", "server", "trame", "none"]

backends = ["ipyvtklink"]

for b in backends:
    print("Using backend: ", b)

    pv.set_jupyter_backend(b)


    # Create a simple mesh (a cube)
    mesh = pv.Cube()

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the mesh to the plotter
    plotter.add_mesh(mesh)

    # Set up the rendering environment
    plotter.show()

# %%

from vedo import dataurl, Mesh, Plotter, Volume, settings

settings.default_backend = 'k3d'

msh = Mesh(dataurl+"beethoven.ply").c('gold').subdivide()
plt = Plotter(bg='black')
plt.show(msh)

# %%


import vedo
import numpy as np

# Define simulation parameters
Nx, Ny, Nz = 30, 30, 30  # Number of spatial points in x, y, and z directions
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Spatial domain dimensions
alpha = 0.01  # Thermal diffusivity
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)  # Spatial steps
dt = 0.01  # Time step

# Initialize temperature field
u = np.zeros((Nx, Ny, Nz))

# Create a vedo Plotter
plotter = vedo.Plotter(title="3D Heat Equation Simulation")

# Function to update the temperature field
def update_temperature(u):
    u_new = u.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                Laplacian = (
                    (u[i+1, j, k] - 2*u[i, j, k] + u[i-1, j, k]) / dx**2 +
                    (u[i, j+1, k] - 2*u[i, j, k] + u[i, j-1, k]) / dy**2 +
                    (u[i, j, k+1] - 2*u[i, j, k] + u[i, j, k-1]) / dz**2
                )
                u_new[i, j, k] = u[i, j, k] + alpha * Laplacian * dt
    return u_new

# Create a vedo Volume object
volume = vedo.Volume(u)

# Add the volume to the plotter
plotter.add(volume)

# Define the simulation time
num_steps = 100
for step in range(num_steps):
    # Update the temperature field
    u = update_temperature(u)
    # Update the volume data
    volume.data = u
    # Add a time label to the plot
    time_label = vedo.Text3D(f"Time Step: {step}", pos=(0, 0, -0.1), s=0.1, c="black")
    # Add the time label to the plotter
    plotter.add(time_label)
    # Capture a screenshot
    vedo.screenshot(f"frames/frame{step:04d}.png")

# Show the animation (if needed)
plotter.show()

# Create an animated GIF from the frames (you can use an external tool)
# Example: convert frames/frame*.png animation.gif

# %%
import vedo
import numpy as np
from scipy.integrate import odeint

# Define simulation parameters
Nx, Ny, Nz = 30, 30, 30  # Number of spatial points in x, y, and z directions
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Spatial domain dimensions
alpha = 0.01  # Thermal diffusivity
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)  # Spatial steps

# Create a 3D grid for spatial points
x, y, z = np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny), np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Initial condition for temperature (e.g., Gaussian distribution)
u0 = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2) / 0.05)

# Flatten the initial condition to a 1D array
u0 = u0.ravel()

# Time points
t = np.linspace(0, 1.0, 100)

# Create a "vedo" plotter
plotter = vedo.Plotter(title="3D Heat Equation Visualization")

# Function to calculate the temperature change over time
def heat_equation(u, t):
    u_new = u.copy()
    u_new = u_new.reshape((Nx, Ny, Nz))
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                Laplacian = (
                    (u_new[i+1, j, k] - 2*u_new[i, j, k] + u_new[i-1, j, k]) / dx**2 +
                    (u_new[i, j+1, k] - 2*u_new[i, j, k] + u_new[i, j-1, k]) / dy**2 +
                    (u_new[i, j, k+1] - 2*u_new[i, j, k] + u_new[i, j, k-1]) / dz**2
                )
                u_new[i, j, k] = u_new[i, j, k] + alpha * Laplacian
    return u_new.ravel()

# Integrate the heat equation over time using odeint
simulation_data = odeint(heat_equation, u0, t)

# Customize visualization settings if needed
cmap = "coolwarm"
alpha = 0.8

# Visualize the temperature field at a specific time step
time_step_to_visualize = 50

# Create a "vedo" object for the selected time step
volume = vedo.Volume(simulation_data[time_step_to_visualize].reshape((Nx, Ny, Nz)))

# Apply visualization settings
volume.cmap(cmap)
volume.alpha(alpha)

# Add the volume to the plotter
plotter.add(volume)

# Show the visualization
plotter.show()

# %%

import vedo

# Define simulation parameters
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Spatial domain dimensions
Nx, Ny, Nz = 10, 10, 10  # Number of points in x, y, and z directions

# Create a "vedo" plotter
plotter = vedo.Plotter(title="3D Domain with Points Visualization")

# Create a 3D representation of the spatial domain (box)
domain_mesh = vedo.Box(pos=[Lx/2, Ly/2, Lz/2], length=Lx, width=Ly, height=Lz, c="lightgray", alpha=0.5)

# Create a mesh grid of points within the box
x = [Lx * i / (Nx - 1) for i in range(Nx)]
y = [Ly * j / (Ny - 1) for j in range(Ny)]
z = [Lz * k / (Nz - 1) for k in range(Nz)]

points = []
for xi in x:
    for yj in y:
        for zk in z:
            points.append([xi, yj, zk])

# Create a 3D representation of all the points
points_mesh = vedo.Points(points, r=3, c="red")

# Add the domain mesh and points mesh to the plotter
plotter.add(domain_mesh)
plotter.add(points_mesh)

# Show the visualization of the 3D domain with all the points
plotter.show()



# %%

import vedo

# Define simulation parameters
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Spatial domain dimensions
Nx, Ny, Nz = 10, 10, 10  # Number of points in x, y, and z directions

# Create a vedo plotter
plotter = vedo.Plotter(title="3D Domain with Points Animation")

# Define the animation function
def animate_color(frame):
    # Create a 3D representation of the spatial domain (box)
    domain_mesh = vedo.Box(pos=[Lx/2, Ly/2, Lz/2], length=Lx, width=Ly, height=Lz, c="lightgray", alpha=0.5)

    # Create a mesh grid of points within the box
    x = [Lx * i / (Nx - 1) for i in range(Nx)]
    y = [Ly * j / (Ny - 1) for j in range(Ny)]
    z = [Lz * k / (Nz - 1) for k in range(Nz)]

    points = []
    for xi in x:
        for yj in y:
            for zk in z:
                points.append([xi, yj, zk])

    # Calculate the color for the point based on the frame number
    if frame > 0:  # Ensure there is more than one frame
        color_value = frame / 5.0  # Linear interpolation from 0 to 1
    else:
        color_value = 0.0

    color = vedo.color_map(color_value, vmin=0, vmax=1)

    # Create a 3D representation of all the points with the current color
    points_mesh = vedo.Points(points, r=3, c=color)

    # Add the domain mesh and points mesh to the plotter
    plotter.add(domain_mesh)
    plotter.add(points_mesh)

# Create the animation frames
for frame in range(50):  # Adjust the number of frames as needed
    animate_color(frame)

# Show the animation in vedo
plotter.show(interactive=True)

# %%
import vedo

# Create a vedo plotter
plotter = vedo.Plotter()

# Create a 3D cube
cube = vedo.Cube(side=0.2)

# Add the cube to the plotter
plotter.add(cube)

# Define a function to rotate the cube
def rotate_cube(event):
    cube.rotateX(1)
    cube.rotateY(1)
    plotter.render()

# Add a timer to continuously call the rotate_cube function
plotter.addCallback("timer", rotate_cube)

# Start the timer (adjust the speed by changing the time interval)
plotter.timer_callback = plotter.timerCallback(timeinterval=10)

# Show the vedo plotter
plotter.show()

# %%

import vedo
import numpy as np

# Create a vedo plotter
plotter = vedo.Plotter()

# Create a grid of points
x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
z = np.sin(np.sqrt(x**2 + y**2))

# Flatten the coordinates
x = x.flatten()
y = y.flatten()
z = z.flatten()

# Create connections between points to simulate a grid-like structure
connections = []
for i in range(9):
    for j in range(9):
        p1 = i * 10 + j
        p2 = p1 + 1
        p3 = (i + 1) * 10 + j
        p4 = p3 + 1
        connections.extend([(p1, p2, p4, p3, p1)])

# Create a grid mesh from the connections
grid_mesh = vedo.Line(points=[(x[i], y[i], z[i]) for i in range(len(x))], c="blue")
grid_mesh.lines(connections)

# Add the grid mesh to the plotter
plotter.add(grid_mesh)

# Show the vedo plotter
plotter.show()


# %%


"""Animated plot showing multiple temporal data lines"""
# Copyright (c) 2021, Nicolas P. Rougier. License: BSD 2-Clause*
# Adapted for vedo by M. Musy, February 2021
from vedo import settings, Line, show
import numpy as np

settings.default_font = "Theemim"

# Generate random data
np.random.seed(1)
data = np.random.uniform(0, 1, (25, 100))
X = np.linspace(-1, 1, data.shape[-1])
G = 0.15 * np.exp(-4 * X**2) # use a  gaussian as a weight

# Generate line plots
lines = []
for i, d in enumerate(data):
    pts = np.c_[X, np.zeros_like(X)+i/10, G*d]
    lines.append(Line(pts, lw=3))

# Set up the first frame
axes = dict(xtitle=':Deltat /:mus', ytitle="source", ztitle="")
plt = show(lines, __doc__, axes=axes, elevation=-30, interactive=False, bg='k8')

for i in range(50):
    data[:, 1:] = data[:, :-1]                      # Shift data to the right
    data[:, 0] = np.random.uniform(0, 1, len(data)) # Fill-in new values
    for line, d in zip(lines, data):                    # Update data
        newpts = line.points()
        newpts[:,2] = G * d
        line.points(newpts).cmap('gist_heat_r', newpts[:,2])
    plt.render()

plt.interactive().close()

# %%
