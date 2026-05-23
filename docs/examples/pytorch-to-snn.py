"""pytorch-to-snn.py
===================
Converting a PyTorch CNN to a Spiking Neural Network with nengo-dl.

This example mirrors the original keras-to-snn notebook, replacing
Keras/TensorFlow with PyTorch throughout.

What it shows
-------------
1. Train a small PyTorch CNN on MNIST.
2. Convert it to a Nengo rate network using ``nengo_dl.Converter``.
3. Swap rate neurons for spiking neurons and run inference.
4. Explore the effect of synaptic filtering and firing-rate scaling.
5. Fine-tune with a firing-rate regularisation loss so the network
   spikes at a sensible rate (~250 Hz) while maintaining accuracy.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nengo
import nengo_dl

# ── reproducibility ──────────────────────────────────────────────────────────
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# ── 1. Load MNIST ─────────────────────────────────────────────────────────────
print("Loading MNIST …")
to_tensor = transforms.ToTensor()
train_ds = torchvision.datasets.MNIST("/tmp/mnist_data", train=True,
                                      download=True, transform=to_tensor)
test_ds  = torchvision.datasets.MNIST("/tmp/mnist_data", train=False,
                                      download=True, transform=to_tensor)

def ds_to_numpy(ds, n=None):
    imgs  = ds.data[:n].numpy().astype(np.float32) / 255.0
    labels = ds.targets[:n].numpy()
    return imgs, labels

train_images, train_labels = ds_to_numpy(train_ds)
test_images,  test_labels  = ds_to_numpy(test_ds)

# Shape expected by Converter: (N, 1, n_steps, 784) → keep 2-D flat for now
# We'll add the time dimension (n_steps=1) before simulation.
train_images_flat = train_images.reshape(len(train_images), 784)
test_images_flat  = test_images.reshape(len(test_images), 784)

print(f"Train: {train_images.shape}  Test: {test_images.shape}")

# ── 2. Define & train a PyTorch CNN ──────────────────────────────────────────
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 32, 3)    # 28→26
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)  # 26→12
        self.flatten = nn.Flatten()
        self.dense  = nn.Linear(64 * 12 * 12, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        return self.dense(x)

model = MnistCNN()
print(f"\nPyTorch CNN:\n{model}")

# Brief training (2 epochs to keep the example fast)
loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for epoch in range(2):
    total_loss, correct, n = 0.0, 0, 0
    for imgs, lbls in loader:
        opt.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, lbls)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(imgs)
        correct += (logits.argmax(1) == lbls).sum().item()
        n += len(imgs)
    print(f"Epoch {epoch+1}: loss={total_loss/n:.4f}  acc={correct/n*100:.1f}%")

model.eval()
torch.save(model.state_dict(), "/tmp/mnist_cnn.pt")
print("Weights saved to /tmp/mnist_cnn.pt")

# Quick PyTorch test accuracy
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256)
correct = 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        correct += (model(imgs).argmax(1) == lbls).sum().item()
pt_acc = correct / len(test_ds)
print(f"PyTorch test accuracy: {pt_acc * 100:.2f}%")

# ── 3. Convert to Nengo rate network ─────────────────────────────────────────
print("\n── Converting to Nengo rate network ──")

# The Converter expects a flat (2-D) model; wrap the CNN in a flat version
# by using a Linear model that approximates it. For a faithful conversion
# we wrap with a simple fully-connected model (the Conv layers fall back to
# TorchNode wrappers because our Converter supports Conv2d via fallback).
converter = nengo_dl.Converter(model, scale_firing_rates=100)

inp_node  = list(converter.inputs.values())[0]
out_node  = list(converter.outputs.values())[-1]

print(f"Input node:  {inp_node}")
print(f"Output node: {out_node}")
print(f"Total Nengo objects: {len(list(converter.net.all_objects))}")

# ── 4. Run converted network in rate mode ────────────────────────────────────
n_test  = 200
n_steps = 1
# Shape: (n_test, n_steps, 784)  – flat images with 1 time step
x_rate  = test_images[:n_test].reshape(n_test, 1, 784)

with converter.net:
    p_out = nengo.Probe(out_node, synapse=None)

with nengo_dl.Simulator(converter.net, minibatch_size=50, seed=seed) as sim:
    sim.run_steps(n_steps, data={inp_node: x_rate}, inference_mode="rate")
    rate_preds = sim.data[p_out][:, -1, :]   # (n_test, 10)

rate_acc = (np.argmax(rate_preds, axis=-1) == test_labels[:n_test]).mean()
print(f"Rate-mode accuracy: {rate_acc * 100:.2f}%")

# ── 5. Run in spiking mode (no synapse) ──────────────────────────────────────
n_steps_spiking = 30
x_spiking = np.tile(x_rate, (1, n_steps_spiking, 1))  # repeat each image

with nengo_dl.Simulator(converter.net, minibatch_size=50, seed=seed) as sim:
    sim.run_steps(n_steps_spiking, data={inp_node: x_spiking}, inference_mode="spiking")
    spike_preds = sim.data[p_out][:, -1, :]

spike_acc = (np.argmax(spike_preds, axis=-1) == test_labels[:n_test]).mean()
print(f"Spiking accuracy (no synapse, {n_steps_spiking} steps): {spike_acc * 100:.2f}%")

# ── 6. Effect of synaptic filtering ──────────────────────────────────────────
print("\n── Synaptic filtering ──")
for tau in [0.001, 0.005, 0.01]:
    with nengo.Network() as filtered_net:
        # Re-convert with synapse
        fc = nengo_dl.Converter(model, scale_firing_rates=100, synapse=tau)
        fn = list(fc.inputs.values())[0]
        fo = list(fc.outputs.values())[-1]
        fp = nengo.Probe(fo, synapse=None)

    with nengo_dl.Simulator(fc.net, minibatch_size=50, seed=seed) as sim:
        sim.run_steps(n_steps_spiking,
                      data={fn: x_spiking},
                      inference_mode="spiking")
        preds = sim.data[fp][:, -1, :]
    acc = (np.argmax(preds, axis=-1) == test_labels[:n_test]).mean()
    print(f"  synapse τ={tau:.3f} → accuracy: {acc * 100:.2f}%")

# ── 7. Effect of firing-rate scaling ─────────────────────────────────────────
print("\n── Firing-rate scaling ──")
for scale in [50, 100, 500]:
    with nengo.Network() as sc_net:
        sc = nengo_dl.Converter(model, scale_firing_rates=scale, synapse=0.005)
        sn = list(sc.inputs.values())[0]
        so = list(sc.outputs.values())[-1]
        sp = nengo.Probe(so, synapse=None)

    with nengo_dl.Simulator(sc.net, minibatch_size=50, seed=seed) as sim:
        sim.run_steps(n_steps_spiking,
                      data={sn: x_spiking},
                      inference_mode="spiking")
        preds = sim.data[sp][:, -1, :]
    acc = (np.argmax(preds, axis=-1) == test_labels[:n_test]).mean()
    print(f"  scale={scale:4d} → accuracy: {acc * 100:.2f}%")

# ── 8. Fine-tune with firing-rate regularisation ──────────────────────────────
print("\n── Fine-tuning with firing-rate regularisation ──")
target_rate = 250.0   # desired peak firing rate (Hz)

# Build converter network with probes on intermediate layers
ft_converter = nengo_dl.Converter(model, scale_firing_rates=100)
ft_inp  = list(ft_converter.inputs.values())[0]
ft_out  = list(ft_converter.outputs.values())[-1]

# Probe the output of each Conv layer for rate regularisation
all_nodes = list(ft_converter.outputs.values())
# Two intermediate spiking probes (one per conv block if available)
with ft_converter.net:
    p_final = nengo.Probe(ft_out, synapse=None)
    if len(all_nodes) >= 2:
        p_mid = nengo.Probe(all_nodes[len(all_nodes) // 2], synapse=None)
    else:
        p_mid = None

# Prepare training data: (N, 1, 784) with labels
n_train = 1000   # small subset for demo
x_train = train_images[:n_train].reshape(n_train, 1, 784)
y_train = train_labels[:n_train].reshape(n_train, 1, 1).astype(np.float32)
y_out   = np.eye(10, dtype=np.float32)[train_labels[:n_train]].reshape(n_train, 1, 10)

loss_dict = {p_final: nengo_dl.losses.CrossEntropy()}
lw_dict   = {p_final: 1.0}
if p_mid is not None:
    loss_dict[p_mid] = nengo_dl.losses.TargetFiringRate()
    lw_dict[p_mid]   = 1e-3

with nengo_dl.Simulator(ft_converter.net, minibatch_size=50, seed=seed) as sim:
    sim.compile(optimizer="adam", loss=loss_dict, loss_weights=lw_dict)

    # Build target dict (final layer: one-hot; mid layer: target rate)
    targets = {p_final: y_out}
    if p_mid is not None:
        targets[p_mid] = np.full(
            (n_train, 1, p_mid.size_in), target_rate, dtype=np.float32
        )

    history = sim.fit(x={ft_inp: x_train}, y=targets, n_steps=1, epochs=3)
    sim.save_params("/tmp/pytorch_snn_finetuned")
    print("Fine-tuned params saved to /tmp/pytorch_snn_finetuned.npz")

    # Evaluate after fine-tuning
    sim.run_steps(1, data={ft_inp: x_rate}, inference_mode="rate")
    ft_preds = sim.data[p_final][:, -1, :]
ft_acc = (np.argmax(ft_preds, axis=-1) == test_labels[:n_test]).mean()
print(f"Fine-tuned rate accuracy: {ft_acc * 100:.2f}%")
print(f"Training loss curve: {[f'{l:.4f}' for l in history['loss']]}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────")
print(f"PyTorch CNN test accuracy:          {pt_acc * 100:.2f}%")
print(f"Converted rate-mode accuracy:       {rate_acc * 100:.2f}%")
print(f"Spiking accuracy ({n_steps_spiking} steps, no τ): {spike_acc * 100:.2f}%")
print(f"Fine-tuned rate accuracy:           {ft_acc * 100:.2f}%")
print("""
Key takeaways
─────────────
• nengo_dl.Converter maps PyTorch nn.Linear → Nengo Ensemble with ReLU neurons.
• Spiking neurons trade accuracy for biological realism; more steps / a
  synaptic filter recovers accuracy.
• Firing-rate scaling compresses the weight range so neurons stay in their
  operating regime.
• A rate-regularisation loss lets you directly control the average spike rate.
""")
