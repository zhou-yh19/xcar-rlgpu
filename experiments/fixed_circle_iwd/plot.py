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

fig, axs = plt.subplots(6, 1, sharex=True)
axs[0].plot([es[7] for (s, u, es, _) in data])
axs[0].set_ylabel(r"$\beta$")

axs[1].plot([es[6] for (s, u, es, _) in data])
axs[1].set_ylabel(r"$V$")

axs[2].plot([u[0] for (s, u, es, _) in data])
axs[2].set_ylabel(r"$\delta$")

axs[3].plot([u[1] for (s, u, es, _) in data], label="f")
axs[3].plot([u[2] for (s, u, es, _) in data], label="r")
axs[3].legend()
axs[3].set_ylabel(r"$\omega$")

axs[3].set_xlabel("Step")

axs[4].plot([es[16] for (s, u, es, _) in data], label="f")
axs[4].plot([es[17] for (s, u, es, _) in data], label="r")
axs[4].set_ylabel(r"$s$")
# axs[4].set_ylim(0, 2)
axs[4].legend()

axs[5].plot([es[12] for (s, u, es, _) in data], label="f")
axs[5].plot([es[14] for (s, u, es, _) in data], label="r")
axs[5].set_ylabel(r"$s_x$")
# axs[5].set_ylim(-2, 1)
axs[5].legend()

axs[5].set_xlabel("Step")
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
ax.plot(radius * np.cos(theta), radius + radius * np.sin(theta), 'r--')
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

