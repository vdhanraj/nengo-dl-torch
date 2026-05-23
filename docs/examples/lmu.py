"""lmu.py
========
Legendre Memory Unit (LMU) on Permuted Sequential MNIST.

This example mirrors the original nengo-dl LMU notebook, replacing the
TensorFlow backend with our PyTorch-based nengo-dl.

The LMU (Voelker et al., 2019) solves psMNIST by treating each image as a
784-timestep sequence (one pixel per step, in a fixed random order).  The
network uses:

  * a mathematically derived linear memory (m) updated by matrix A and B,
  * a learned non-linear hidden layer (h) that reads from x and m,
  * a dense linear readout over the last hidden state.

The A/B matrices encode a sliding-window orthogonal basis (Legendre
polynomials) so that the memory can be perfectly reconstructed from the
compressed state.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nengo
from nengo.utils.filter_design import cont2discrete
import nengo_dl

# ── reproducibility ───────────────────────────────────────────────────────────
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.RandomState(seed)

# ── 1. Load & prepare psMNIST ─────────────────────────────────────────────────
print("Loading MNIST …")
to_tensor = transforms.ToTensor()
train_ds = torchvision.datasets.MNIST("/tmp/mnist_data", train=True,
                                      download=True, transform=to_tensor)
test_ds  = torchvision.datasets.MNIST("/tmp/mnist_data", train=False,
                                      download=True, transform=to_tensor)

train_images = train_ds.data.numpy().astype(np.float32) / 255.0
test_images  = test_ds.data.numpy().astype(np.float32) / 255.0
train_labels = train_ds.targets.numpy()
test_labels  = test_ds.targets.numpy()

# Flatten 28×28 → 784, add a dummy feature dim: (N, 784, 1)
train_images = train_images.reshape(len(train_images), -1, 1)
test_images  = test_images.reshape(len(test_images),  -1, 1)

# Apply a fixed random permutation to the pixel order
perm = rng.permutation(train_images.shape[1])
train_images = train_images[:, perm, :]
test_images  = test_images[:, perm, :]

# nengo-dl expects targets shaped (N, n_steps, out_dim).
# Because we probe only the final timestep (keep_history=False),
# reshape labels to (N, 1, 1).
train_labels_nd = train_labels[:, None, None]
test_labels_nd  = test_labels[:, None, None]

print(f"Train images: {train_images.shape}  labels: {train_labels_nd.shape}")
print(f"Test  images: {test_images.shape}   labels: {test_labels_nd.shape}")

# Show one example
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(train_ds.data[0].numpy(), cmap="gray")
axes[0].set_title(f"Original image  (label={train_labels[0]})")
axes[0].axis("off")
axes[1].imshow(train_images[0].reshape(8, -1), cmap="gray")
axes[1].set_title("Permuted pixel sequence (8 rows)")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("/tmp/lmu_input.png", dpi=100)
print("Input visualisation saved to /tmp/lmu_input.png")

# ── 2. LMU cell as a Nengo Network ───────────────────────────────────────────
class LMUCell(nengo.Network):
    """One LMU cell: linear memory m_t and learned hidden state h_t.

    Parameters
    ----------
    units : int
        Dimensionality of the hidden state h.
    order : int
        Order of the Legendre polynomial basis (size of memory m).
    theta : int
        Window length (number of timesteps to remember).
    input_d : int
        Dimensionality of the input x at each timestep.
    """

    def __init__(self, units, order, theta, input_d, **kwargs):
        super().__init__(**kwargs)

        # ── compute A, B via ZOH discretisation ──────────────────────────
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))
        A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")

        with self:
            nengo_dl.configure_settings(trainable=None)

            # ── network nodes ────────────────────────────────────────────
            self.x = nengo.Node(size_in=input_d)    # input at this step
            self.u = nengo.Node(size_in=1)           # scalar encoding of x
            self.m = nengo.Node(size_in=order)       # Legendre memory

            # h: learned hidden state; use tanh via a TorchNode
            tanh_module = nn.Tanh()
            self.h = nengo_dl.TorchNode(
                tanh_module, size_in=units, size_out=units, pass_time=False
            )

            # ── u_t  (project input to scalar) ───────────────────────────
            nengo.Connection(self.x, self.u,
                             transform=np.ones((1, input_d)), synapse=None)

            # ── m_t  (fixed A, B — not trained) ──────────────────────────
            conn_A = nengo.Connection(self.m, self.m, transform=A, synapse=0)
            conn_B = nengo.Connection(self.u, self.m, transform=B, synapse=None)
            self.config[conn_A].trainable = False
            self.config[conn_B].trainable = False

            # ── h_t  (trainable connections from x, h_{t-1}, m_t) ────────
            nengo.Connection(self.x, self.h,
                             transform=nengo_dl.dists.Glorot(), synapse=None)
            nengo.Connection(self.h, self.h,
                             transform=nengo_dl.dists.Glorot(), synapse=0)
            nengo.Connection(self.m, self.h,
                             transform=nengo_dl.dists.Glorot(), synapse=None)


# ── 3. Full network ───────────────────────────────────────────────────────────
units  = 212   # hidden dimension
order  = 256   # memory order
theta  = train_images.shape[1]   # = 784 — remember the whole sequence

with nengo.Network(seed=seed) as net:
    nengo_dl.configure_settings(
        trainable=None,
        stateful=False,
        keep_history=False,   # only keep the last timestep
    )

    # input node (1 pixel per timestep)
    inp = nengo.Node(np.zeros(train_images.shape[-1]))

    # LMU cell
    lmu = LMUCell(units=units, order=order, theta=theta,
                  input_d=train_images.shape[-1])
    conn_inp = nengo.Connection(inp, lmu.x, synapse=None)
    net.config[conn_inp].trainable = False

    # dense linear readout from h → 10 classes
    out = nengo.Node(size_in=10)
    nengo.Connection(lmu.h, out,
                     transform=nengo_dl.dists.Glorot(), synapse=None)

    # probe only the final timestep (keep_history=False ensures this)
    p = nengo.Probe(out)

print(f"\nLMU network: units={units}, order={order}, theta={theta}")
print(f"Trainable connections: x→h, h→h, m→h, h→out")
print(f"Fixed connections: m→m (A), u→m (B)")

# ── 4. Train ──────────────────────────────────────────────────────────────────
# The sequence length is 784, so unroll_simulation controls memory usage.
# We use a small subset of training data to keep this example fast.
n_train = 5000
n_test  = 1000
mini    = 100

# One target per input (at the last timestep); shape (N, 1, 1)
train_y = train_labels_nd[:n_train]
test_y  = test_labels_nd[:n_test]
train_x = train_images[:n_train]
test_x  = test_images[:n_test]

print(f"\nTraining on {n_train} examples, minibatch={mini}, epochs=2 …")
print("(For the full result from the paper, train on 60 000 examples for 10 epochs)")

with nengo_dl.Simulator(net, minibatch_size=mini, seed=seed) as sim:
    sim.compile(
        optimizer="adam",
        loss={p: nengo_dl.losses.CrossEntropy()},
    )

    # Evaluate before training
    sim.run_steps(train_images.shape[1],
                  data={inp: test_x},
                  inference_mode="rate")
    pre_preds = np.argmax(sim.data[p], axis=-1).ravel()[:n_test]
    pre_acc = (pre_preds == test_labels[:n_test]).mean()
    print(f"Accuracy before training: {pre_acc * 100:.2f}%")

    history = sim.fit(
        x={inp: train_x},
        y={p: np.eye(10, dtype=np.float32)[train_labels[:n_train]].reshape(n_train, 1, 10)},
        n_steps=train_images.shape[1],
        epochs=2,
    )
    sim.save_params("/tmp/lmu_params")
    print("Params saved to /tmp/lmu_params.npz")

    # Evaluate after training
    sim.run_steps(test_images.shape[1],
                  data={inp: test_x},
                  inference_mode="rate")
    post_preds = np.argmax(sim.data[p], axis=-1).ravel()[:n_test]
    post_acc = (post_preds == test_labels[:n_test]).mean()
    print(f"Accuracy after training:  {post_acc * 100:.2f}%")

# ── 5. Plot training loss ─────────────────────────────────────────────────────
plt.figure(figsize=(6, 3))
plt.plot(history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("LMU training loss (psMNIST)")
plt.tight_layout()
plt.savefig("/tmp/lmu_loss.png", dpi=100)
print("Loss curve saved to /tmp/lmu_loss.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("""
Key takeaways
─────────────
• The LMU uses a mathematically derived memory (A, B from Legendre polynomials)
  to compress past context without needing to learn these connections.
• The hidden layer h learns to combine the current input, its own previous
  state, and the compressed memory to classify 784-step sequences.
• synapse=0 on recurrent connections adds a one-timestep delay, making the
  network truly recurrent across simulation steps.
• nengo_dl.configure_settings(keep_history=False) tells the simulator to only
  keep the last probe value, reducing memory usage for long sequences.
""")
