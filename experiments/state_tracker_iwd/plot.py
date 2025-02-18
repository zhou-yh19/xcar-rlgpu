from matplotlib import pyplot as plt
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

filename = latest_file if args.filename is None else args.filename

fig, axs = plt.subplots(5, 1, sharex=True)
axs[0].plot([obs[0] for (s, u, es, obs) in data], label="Actual")
axs[0].plot([obs[4] for (s, u, es, obs) in data], label="Ref")
axs[0].legend()
axs[0].set_ylabel(r"$r$")

axs[1].plot([obs[1] for (s, u, es, obs) in data], label="Actual")
axs[1].plot([obs[5] for (s, u, es, obs) in data], label="Ref")
axs[1].legend()
axs[1].set_ylabel(r"$\beta$")

axs[2].plot([obs[2] for (s, u, es, obs) in data], label="Actual")
axs[2].plot([obs[6] for (s, u, es, obs) in data], label="Ref")
axs[2].legend()
axs[2].set_ylabel(r"$V$")

axs[3].plot([u[0] for (s, u, es, obs) in data])
axs[3].set_ylabel(r"$\delta$")

axs[4].plot([u[1] for (s, u, es, obs) in data], label="f")
axs[4].plot([u[2] for (s, u, es, obs) in data], label="r")
axs[4].legend()
axs[4].set_ylabel(r"$\omega$")

axs[4].set_xlabel("Step")

plt.savefig(f"{filename}_state.png")

# Plot the trajectory
fig, ax = plt.subplots(1, 1)
theta = np.linspace(0, 2 * np.pi, 100)
x = [s[0] for (s, u, es, _) in data]
y = [s[1] for (s, u, es, _) in data]
psi = [s[2] for (s, u, es, _) in data]

# Plot the path of the car
radius = 1.
ax.plot(x, y)
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

plt.savefig(f"{filename}_trajectory.png")