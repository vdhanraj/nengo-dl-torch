"""pytorch-models.py
===================
Integrating PyTorch Models with Nengo via nengo-dl.

This example mirrors the original tensorflow-models notebook, replacing
Keras/TensorFlow with PyTorch throughout.

It demonstrates four ways to combine PyTorch and Nengo in a single simulation:

1. **TorchNode** – wrap an arbitrary ``nn.Module`` as a Nengo Node.
2. **nengo_dl.Layer** – Keras-style functional API for building networks
   inline without leaving the ``with nengo.Network()`` block.
3. **Manual Nengo equivalent** – build a linear→ReLU block using plain Nengo
   ``Ensemble`` and ``Connection`` objects.
4. **nengo_dl.Converter** – automatically convert a trained PyTorch model
   into a full Nengo network ready for spiking inference.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
%matplotlib inline
import matplotlib.pyplot as plt
import nengo
import nengo_dl

# ── reproducibility ───────────────────────────────────────────────────────────
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# ── Fashion MNIST class names ─────────────────────────────────────────────────
CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load Fashion MNIST
# ─────────────────────────────────────────────────────────────────────────────
print("Loading Fashion MNIST …")
to_tensor = transforms.ToTensor()
train_ds = torchvision.datasets.FashionMNIST(
    "/tmp/fashion_mnist", train=True,  download=True, transform=to_tensor
)
test_ds  = torchvision.datasets.FashionMNIST(
    "/tmp/fashion_mnist", train=False, download=True, transform=to_tensor
)

train_images = train_ds.data.numpy().astype(np.float32) / 255.0
test_images  = test_ds.data.numpy().astype(np.float32)  / 255.0
train_labels = train_ds.targets.numpy()
test_labels  = test_ds.targets.numpy()

# Flatten 28×28 → 784
train_images_flat = train_images.reshape(len(train_images), 784)
test_images_flat  = test_images.reshape(len(test_images), 784)

print(f"Train: {train_images.shape}   Test: {test_images.shape}")

# Quick visualisation
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(train_images[i], cmap="gray")
    ax.set_title(CLASS_NAMES[train_labels[i]])
    ax.axis("off")
plt.suptitle("Fashion MNIST samples")
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Demo: TorchNode wrapping an nn.Module
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. TorchNode demo ──")

class NormalisingNet(nn.Module):
    """Subtracts the mean of each sample (centering)."""
    def forward(self, x):
        return x - x.mean(dim=-1, keepdim=True)

n_demo   = 10
n_steps  = 1

with nengo.Network(seed=seed) as demo_net:
    inp   = nengo.Node(np.zeros(784))
    node  = nengo_dl.TorchNode(NormalisingNet(), size_in=784, size_out=784,
                               pass_time=False)
    nengo.Connection(inp, node, synapse=None)
    p_demo = nengo.Probe(node, synapse=None)

x_demo = test_images_flat[:n_demo].reshape(n_demo, 1, 784)

with nengo_dl.Simulator(demo_net, minibatch_size=n_demo, seed=seed) as sim:
    sim.run_steps(n_steps, data={inp: x_demo})
    out_demo = sim.data[p_demo][:, 0, :]

print(f"Input  mean (first sample): {x_demo[0, 0].mean():.4f}")
print(f"Output mean (first sample): {out_demo[0].mean():.6f}  (≈ 0)")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Demo: nengo_dl.Layer functional API
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. nengo_dl.Layer API demo ──")

n_hidden = 128
n_out    = 10

with nengo.Network(seed=seed) as layer_net:
    nengo_dl.configure_settings(trainable=None)

    inp = nengo.Node(np.zeros(784))

    # Stack: Linear(784→128) → ReLU → Linear(128→64) → ReLU → Linear(64→10)
    x = nengo_dl.Layer(nn.Linear(784, n_hidden))(inp)
    x = nengo_dl.Layer(nn.ReLU())(x)
    x = nengo_dl.Layer(nn.Linear(n_hidden, 64))(x)
    x = nengo_dl.Layer(nn.ReLU())(x)
    out_node = nengo_dl.Layer(nn.Linear(64, n_out))(x)
    p_layer = nengo.Probe(out_node, synapse=None)

print(f"Layer network: 784 → {n_hidden} → 64 → 10")

n_train  = 5000
n_test   = 1000
mini     = 100

x_train = train_images_flat[:n_train].reshape(n_train, 1, 784)
y_train = np.eye(10, dtype=np.float32)[train_labels[:n_train]].reshape(n_train, 1, 10)
x_test  = test_images_flat[:n_test].reshape(n_test, 1, 784)

with nengo_dl.Simulator(layer_net, minibatch_size=mini, seed=seed) as sim:
    sim.compile(optimizer="adam", loss={p_layer: nengo_dl.losses.CrossEntropy()})

    history_layer = sim.fit(
        x={inp: x_train},
        y={p_layer: y_train},
        n_steps=1,
        epochs=5,
    )
    sim.save_params("/tmp/pm_layer_params")

    sim.run_steps(1, data={inp: x_test})
    layer_preds = np.argmax(sim.data[p_layer][:, 0, :], axis=-1)

layer_acc = (layer_preds == test_labels[:n_test]).mean()
print(f"Layer-API test accuracy: {layer_acc * 100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Demo: manual Nengo equivalent (Ensemble + Connection)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. Manual Nengo ensemble network ──")

n_ens = 128   # number of neurons ≈ hidden units

with nengo.Network(seed=seed) as ens_net:
    ens_net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    ens_net.config[nengo.Connection].synapse   = None

    inp    = nengo.Node(np.zeros(784))
    hidden = nengo.Ensemble(n_ens, dimensions=1)   # neurons serve as units
    out_nd = nengo.Node(size_in=10)

    nengo.Connection(inp,    hidden.neurons, transform=nengo_dl.dists.Glorot())
    nengo.Connection(hidden.neurons, out_nd, transform=nengo_dl.dists.Glorot())

    p_ens = nengo.Probe(out_nd, synapse=None)

print(f"Manual Nengo: 784 → {n_ens}-neuron Ensemble → 10")

with nengo_dl.Simulator(ens_net, minibatch_size=mini, seed=seed) as sim:
    sim.compile(optimizer="adam", loss={p_ens: nengo_dl.losses.CrossEntropy()})

    history_ens = sim.fit(
        x={inp: x_train},
        y={p_ens: y_train},
        n_steps=1,
        epochs=5,
    )

    sim.run_steps(1, data={inp: x_test})
    ens_preds = np.argmax(sim.data[p_ens][:, 0, :], axis=-1)

ens_acc = (ens_preds == test_labels[:n_test]).mean()
print(f"Manual-Ensemble test accuracy: {ens_acc * 100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Train a native PyTorch model, then convert with Converter
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. PyTorch model → Converter ──")

class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 128)
        self.fc2  = nn.Linear(128, 64)
        self.out  = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

pt_model = FashionMLP()
print(f"PyTorch MLP:\n{pt_model}")

loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
opt    = torch.optim.Adam(pt_model.parameters(), lr=1e-3)
pt_model.train()
for epoch in range(3):
    total_loss, correct, n = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs_flat = imgs.reshape(len(imgs), 784)
        opt.zero_grad()
        logits = pt_model(imgs_flat)
        loss   = F.cross_entropy(logits, lbls)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == lbls).sum().item()
        n          += len(imgs)
    print(f"Epoch {epoch+1}: loss={total_loss/n:.4f}  acc={correct/n*100:.1f}%")

pt_model.eval()
torch.save(pt_model.state_dict(), "/tmp/pm_fashion_mlp.pt")
print("PyTorch weights saved to /tmp/pm_fashion_mlp.pt")

# Quick PyTorch test accuracy
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256)
correct = 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        correct += (pt_model(imgs.reshape(len(imgs), 784)).argmax(1) == lbls).sum().item()
pt_acc = correct / len(test_ds)
print(f"PyTorch test accuracy: {pt_acc * 100:.2f}%")

# Convert to Nengo
converter = nengo_dl.Converter(pt_model, scale_firing_rates=100)
conv_inp  = list(converter.inputs.values())[0]
conv_out  = list(converter.outputs.values())[-1]

with converter.net:
    p_conv = nengo.Probe(conv_out, synapse=None)

x_conv = test_images_flat[:n_test].reshape(n_test, 1, 784)

with nengo_dl.Simulator(converter.net, minibatch_size=50, seed=seed) as sim:
    sim.run_steps(1, data={conv_inp: x_conv}, inference_mode="rate")
    conv_preds = np.argmax(sim.data[p_conv][:, 0, :], axis=-1)

conv_acc = (conv_preds == test_labels[:n_test]).mean()
print(f"Converter rate-mode accuracy: {conv_acc * 100:.2f}%")

# Also run in spiking mode
n_steps_spike = 30
x_spike = np.tile(x_conv, (1, n_steps_spike, 1))

with nengo_dl.Simulator(converter.net, minibatch_size=50, seed=seed) as sim:
    sim.run_steps(n_steps_spike, data={conv_inp: x_spike}, inference_mode="spiking")
    spike_preds = np.argmax(sim.data[p_conv][:, -1, :], axis=-1)

spike_acc = (spike_preds == test_labels[:n_test]).mean()
print(f"Converter spiking accuracy ({n_steps_spike} steps): {spike_acc * 100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plot training loss curves
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(history_layer["loss"], label="Layer API")
axes[0].plot(history_ens["loss"],   label="Manual Ensemble")
axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].legend()

accs = [layer_acc, ens_acc, conv_acc, spike_acc]
bars = axes[1].bar(
    ["Layer\nAPI", "Manual\nEnsemble", "Converter\n(rate)", "Converter\n(spiking)"],
    [a * 100 for a in accs],
    color=["steelblue", "seagreen", "darkorange", "crimson"],
)
axes[1].set_ylim([0, 100])
axes[1].set_ylabel("Test accuracy (%)")
axes[1].set_title(f"Fashion MNIST – {n_test} test samples")
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
Results (Fashion MNIST – {n_test} test samples)
───────────────────────────────────────────────
nengo_dl.Layer API:           {layer_acc * 100:.2f}%
Manual Nengo Ensemble:        {ens_acc * 100:.2f}%
Converter (rate mode):        {conv_acc * 100:.2f}%   (≈ PyTorch: {pt_acc * 100:.2f}%)
Converter (spiking, {n_steps_spike} steps): {spike_acc * 100:.2f}%

Key takeaways
─────────────
• TorchNode wraps any nn.Module as a Nengo Node — functions are called at
  each simulation timestep with gradient support.
• nengo_dl.Layer gives a Keras-style functional API for building networks
  inside a Nengo context: nn.Linear, nn.ReLU, and NeuronType layers.
• A manual Nengo Ensemble with RectifiedLinear neurons is equivalent to a
  nn.Linear + ReLU layer; nengo-dl trains both paths identically.
• nengo_dl.Converter automatically converts a trained PyTorch model to a
  Nengo network.  Rate mode reproduces PyTorch accuracy; spiking mode trades
  some accuracy for biological realism, which improves with more timesteps.
""")
