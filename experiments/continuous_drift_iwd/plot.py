from matplotlib import pyplot as plt
from pandas import read_csv
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import glob
import os
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default=None)
args = parser.parse_args()

list_of_files = glob.glob('data/*')
latest_file = max(list_of_files, key=os.path.getctime)
data = torch.load(latest_file)

list_of_files_traj = glob.glob('data/traj*')
latest_file_traj = max(list_of_files_traj, key=os.path.getctime)
data_traj = read_csv(latest_file_traj, header=None)

filename = latest_file if args.filename is None else args.filename

# Create the figure and subplots
fig, axs = plt.subplots(11, 1, sharex=True, figsize=(10, 18))
fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.1, right=0.95)

# Plot each variable on its corresponding subplot
axs[0].plot([s[0] for (s, u, es, _) in data], label="x")
axs[0].set_ylabel(r"$x$")

axs[1].plot([s[1] for (s, u, es, _) in data], label="y")
axs[1].set_ylabel(r"$y$")

axs[2].plot([s[2] for (s, u, es, _) in data], label="psi")
axs[2].set_ylabel(r"$\psi$")

axs[3].plot([es[2] for (s, u, es, _) in data])
axs[3].set_ylabel(r"$r$")

axs[4].plot([es[7] for (s, u, es, _) in data])
axs[4].set_ylabel(r"$\beta$")

axs[5].plot([es[6] for (s, u, es, _) in data])
axs[5].set_ylabel(r"$V$")

axs[6].plot([u[0] for (s, u, es, _) in data])
axs[6].set_ylabel(r"$\delta$")

axs[7].plot([u[1] for (s, u, es, _) in data], label="r")
axs[7].plot([u[2] for (s, u, es, _) in data], label="l")
axs[7].set_ylabel(r"$\omega f$")
axs[7].legend()

axs[8].plot([u[3] for (s, u, es, _) in data], label="r")
axs[8].plot([u[4] for (s, u, es, _) in data], label="l")
axs[8].set_ylabel(r"$\omega r$")
axs[8].legend()

axs[9].plot([es[16] for (s, u, es, _) in data], label="r")
axs[9].plot([es[17] for (s, u, es, _) in data], label="l")
axs[9].set_ylabel(r"$sf_x$")
axs[9].legend()

axs[10].plot([es[18] for (s, u, es, _) in data], label="r")
axs[10].plot([es[19] for (s, u, es, _) in data], label="l")
axs[10].set_ylabel(r"$sr_x$")
axs[10].legend()

# Set common X-axis label
axs[10].set_xlabel("Step")

# Save the figure
plt.savefig(f"{filename}_state.png", dpi=300, bbox_inches='tight')

# Plot the trajectory
fig, ax = plt.subplots(1, 1)
theta = np.linspace(0, 2 * np.pi, 100)
x = [s[0] for (s, u, es, _) in data]
y = [s[1] for (s, u, es, _) in data]
psi = [s[2] for (s, u, es, _) in data]

# Plot the path of the car
ax.plot(x, y)
ax.plot(data_traj[0], data_traj[1], 'r--')
ax.set_aspect('equal', 'box')

# Plot the orientation of the car
indices = [i for i in range(0, len(data), 100)]
for idx in indices:
    x_, y_ = x[idx], y[idx]
    psi_ = psi[idx] * 180 / np.pi
    
    length = 0.15
    dx = length * np.cos(np.radians(psi_))
    dy = length * np.sin(np.radians(psi_))
    ax.arrow(x_, y_, dx, dy, head_width=0.08, head_length=0.08, fc='r', ec='r')

# Save the figure
plt.savefig(f"{filename}_trajectory.png")

