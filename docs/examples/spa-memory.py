"""spa-memory.py
===============
Semantic Pointer Architecture (SPA) – Working Memory via Recurrent Dynamics.

This example mirrors the original spa-memory notebook.  It demonstrates:

1. A simple recurrent memory network that holds a semantic pointer over a
   delay period (no binding).
2. A binding-retrieval network: role⊛filler items are stored in a recurrent
   memory, then retrieved by circular correlation with a cue.

Both networks are trained with nengo-dl to improve accuracy.

Outline
-------
a. Simple memory
   - Input: a semantic pointer is presented for ``presentation_time`` seconds,
     then removed (0-vector).
   - Target: the same pointer should be maintained throughout the delay.
   - Network: ``input_node → memory_ensemble`` with a recurrent self-connection
     (``transform=1, synapse=tau``).

b. Binding memory
   - Input: ``n_pairs`` role⊛filler bindings are summed into a trace and fed
     into the memory; after ``presentation_time`` a cue (one of the roles) is
     presented.
   - Target: during the cue phase the memory output should match the
     corresponding filler pointer; NaN targets are used during the binding
     phase (so that loss is not computed there).
   - Network: ``trace → cconv → memory`` and ``(memory, cue) → ccorr → output``.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import nengo
from nengo import spa
import nengo_dl

# ── reproducibility ───────────────────────────────────────────────────────────
seed = 0

# ── shared hyperparameters ────────────────────────────────────────────────────
dims             = 32
tau              = 0.01     # recurrent memory time-constant (seconds)
dt               = 0.001    # simulation timestep
presentation_time = 0.1     # seconds for which the item is presented
delay_time        = 0.1     # seconds of memory to maintain after input removed
n_pairs           = 2       # role/filler pairs per binding example


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data generation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _make_ramp(n_present, n_delay):
    """Ramp from 0→1 over presentation, then stays at 1 during delay."""
    ramp = np.ones(n_present + n_delay)
    ramp[:n_present] = np.linspace(0, 1, n_present)
    return ramp


def get_memory_data(n_inputs, vec_d, vocab_seed, presentation_time, delay_time, dt):
    """Simple memory task: present a pointer, hold it over the delay.

    Returns
    -------
    inputs  : (n_inputs, n_steps, vec_d)  – input sequence (zeros after present)
    outputs : (n_inputs, n_steps, vec_d)  – desired memory content
    vocab   : Vocabulary
    """
    rng  = np.random.RandomState(vocab_seed)
    vocab = spa.Vocabulary(dimensions=vec_d, rng=rng, max_similarity=1)

    n_present = int(presentation_time / dt)
    n_delay   = int(delay_time / dt)
    n_steps   = n_present + n_delay

    ramp = _make_ramp(n_present, n_delay)            # (n_steps,)

    inputs  = np.zeros((n_inputs, n_steps, vec_d), dtype=np.float32)
    outputs = np.zeros((n_inputs, n_steps, vec_d), dtype=np.float32)

    for n in range(n_inputs):
        name = f"MEM_{n}"
        vocab.parse(name)
        ptr = vocab[name].v.astype(np.float32)
        inputs[n, :n_present, :]  = ptr          # present for first n_present steps
        outputs[n, :, :]          = ptr[None, :] * ramp[:, None]

    return inputs, outputs, vocab


def get_binding_data(n_items, pairs_per_item, vec_d, rng_seed,
                     presentation_time, delay_time, dt):
    """Binding task: present trace, cue with a role, retrieve the filler.

    During the presentation phase the target is NaN (we ignore that part of
    the loss).  During the retrieval/delay phase the target is the filler
    pointer.

    Returns
    -------
    traces   : (n_items, n_steps, vec_d)  – cumulative sum of role⊛filler pairs
    cues     : (n_items, n_steps, vec_d)  – zero then cue role during delay
    targets  : (n_items, n_steps, vec_d)  – NaN during presentation, filler during delay
    vocab    : Vocabulary
    """
    rng   = np.random.RandomState(rng_seed)
    vocab = spa.Vocabulary(dimensions=vec_d, rng=rng, max_similarity=1)

    n_present = int(presentation_time / dt)
    n_delay   = int(delay_time / dt)
    n_steps   = n_present + n_delay

    traces  = np.zeros((n_items, n_steps, vec_d), dtype=np.float32)
    cues    = np.zeros((n_items, n_steps, vec_d), dtype=np.float32)
    targets = np.full((n_items, n_steps, vec_d), np.nan, dtype=np.float32)

    for n in range(n_items):
        role_names   = [f"ROLE_{n}_{i}"   for i in range(pairs_per_item)]
        filler_names = [f"FILLER_{n}_{i}" for i in range(pairs_per_item)]

        trace_ptr = vocab.parse(
            "+".join(f"{r} * {f}" for r, f in zip(role_names, filler_names))
        )
        trace_ptr.normalize()

        cue_idx = rng.randint(pairs_per_item)
        cue_v    = vocab[role_names[cue_idx]].v.astype(np.float32)
        filler_v = vocab[filler_names[cue_idx]].v.astype(np.float32)

        traces[n, :n_present, :]  = trace_ptr.v.astype(np.float32)
        cues[n, n_present:, :]    = cue_v
        targets[n, n_present:, :] = filler_v        # NaN during presentation

    return traces, cues, targets, vocab


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NaN-aware MSE loss
# ─────────────────────────────────────────────────────────────────────────────
class NanMSE(torch.nn.Module):
    """MSE that ignores timesteps where target is NaN."""

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        return F.mse_loss(pred[mask], target[mask])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build simple-memory network
# ─────────────────────────────────────────────────────────────────────────────
n_present = int(presentation_time / dt)
n_delay   = int(delay_time / dt)
n_steps   = n_present + n_delay

with nengo.Network(seed=seed) as mem_net:
    mem_net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    mem_net.config[nengo.Connection].synapse   = None

    mem_inp = nengo.Node(np.zeros(dims))

    # Memory ensemble: recurrent self-connection for sustained activity
    memory  = nengo.Ensemble(200, dims)
    # Feed-in: scale by tau / t_integrate so the steady-state ≈ input
    t_integrate = 0.05
    nengo.Connection(mem_inp, memory, transform=tau / t_integrate, synapse=None)
    nengo.Connection(memory, memory, transform=1.0, synapse=tau)

    mem_probe = nengo.Probe(memory, synapse=None)

print(f"Simple-memory network: 200-neuron Ensemble, dims={dims}")
print(f"Simulation: {n_steps} steps ({n_steps * dt:.3f} s)")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Evaluate simple-memory before training
# ─────────────────────────────────────────────────────────────────────────────
n_test  = 50
mini_sz = 50

test_mem_in, test_mem_out, test_vocab = get_memory_data(
    n_test, dims, seed, presentation_time, delay_time, dt
)

with nengo_dl.Simulator(mem_net, minibatch_size=mini_sz, seed=seed) as sim:
    sim.run_steps(n_steps, data={mem_inp: test_mem_in})
    pre_output = sim.data[mem_probe].copy()   # (n_test, n_steps, dims)

# Accuracy: at each delay step, nearest-neighbour in vocabulary
def mem_accuracy(output, vocab, targets, delay_start):
    """Fraction of items where nearest neighbour matches target at end of delay."""
    out  = output[:, -1, :]                        # last step
    tgt  = targets[:, -1, :]
    sims = np.dot(vocab.vectors, out.T)            # (n_vocab, batch)
    idxs = np.argmax(sims, axis=0)
    return np.mean(np.all(vocab.vectors[idxs] == tgt, axis=1))

pre_mem_acc = mem_accuracy(pre_output, test_vocab, test_mem_out, n_present)
print(f"\nSimple-memory accuracy (no training): {pre_mem_acc * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Train simple-memory network
# ─────────────────────────────────────────────────────────────────────────────
print("Training simple-memory network …")
n_train = 2000

train_mem_in, train_mem_out, _ = get_memory_data(
    n_train, dims, seed + 1, presentation_time, delay_time, dt
)

with nengo_dl.Simulator(mem_net, minibatch_size=mini_sz, seed=seed) as sim:
    sim.compile(optimizer="rmsprop", loss={mem_probe: "mse"})
    history_mem = sim.fit(
        x={mem_inp: train_mem_in},
        y={mem_probe: train_mem_out},
        n_steps=n_steps,
        epochs=10,
    )
    sim.save_params("/tmp/spa_memory_params")
    print("Params saved to /tmp/spa_memory_params.npz")

    sim.run_steps(n_steps, data={mem_inp: test_mem_in})
    post_output = sim.data[mem_probe].copy()

post_mem_acc = mem_accuracy(post_output, test_vocab, test_mem_out, n_present)
print(f"Simple-memory accuracy (after training): {post_mem_acc * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Visualise simple memory (one example)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t_axis = np.arange(n_steps) * dt

# similarity over time for first test item
sims_pre  = spa.similarity(pre_output[0], test_vocab)
sims_post = spa.similarity(post_output[0], test_vocab)

axes[0].plot(t_axis, sims_pre)
axes[0].axvline(presentation_time, color="k", linestyle="--", label="end of input")
axes[0].set_ylabel("Cosine similarity")
axes[0].set_title("Memory similarity (before training)")
axes[0].legend(loc="upper right")

axes[1].plot(t_axis, sims_post)
axes[1].axvline(presentation_time, color="k", linestyle="--")
axes[1].set_ylabel("Cosine similarity")
axes[1].set_title("Memory similarity (after training)")
axes[1].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("/tmp/spa_memory_simple.png", dpi=100)
print("Simple-memory plot saved to /tmp/spa_memory_simple.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Build binding + retrieval network
# ─────────────────────────────────────────────────────────────────────────────
with nengo.Network(seed=seed) as bind_net:
    bind_net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    bind_net.config[nengo.Connection].synapse   = None

    trace_inp = nengo.Node(np.zeros(dims))
    cue_inp   = nengo.Node(np.zeros(dims))

    # Encode: bind trace into memory via cconv
    cconv = nengo.networks.CircularConvolution(5, dims, invert_b=False)
    nengo.Connection(trace_inp, cconv.input_a)
    nengo.Connection(trace_inp, cconv.input_b)

    # Memory: hold the bound trace
    bind_mem = nengo.Ensemble(200, dims)
    nengo.Connection(cconv.output, bind_mem,
                     transform=tau / t_integrate, synapse=None)
    nengo.Connection(bind_mem, bind_mem, transform=1.0, synapse=tau)

    # Decode: circular correlation of memory with cue
    ccorr = nengo.networks.CircularConvolution(5, dims, invert_b=True)
    nengo.Connection(bind_mem, ccorr.input_a)
    nengo.Connection(cue_inp,  ccorr.input_b)

    bind_probe  = nengo.Probe(bind_mem, synapse=None)
    out_probe   = nengo.Probe(ccorr.output, synapse=None)

print(f"\nBinding network: cconv + memory + ccorr, dims={dims}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Evaluate binding network before training
# ─────────────────────────────────────────────────────────────────────────────
n_bind_test = 50

test_traces, test_cues, test_bind_targets, bind_vocab = get_binding_data(
    n_bind_test, n_pairs, dims, seed, presentation_time, delay_time, dt
)

bind_inputs = {trace_inp: test_traces, cue_inp: test_cues}

with nengo_dl.Simulator(bind_net, minibatch_size=mini_sz, seed=seed) as sim:
    sim.run_steps(n_steps, data=bind_inputs)
    pre_bind_out = sim.data[out_probe].copy()

def bind_accuracy(output, vocab, targets, start_step):
    """Accuracy at last timestep (filler retrieval phase)."""
    out  = output[:, -1, :]
    tgt  = targets[:, -1, :]
    mask = ~np.any(np.isnan(tgt), axis=1)
    if mask.sum() == 0:
        return 0.0
    sims = np.dot(vocab.vectors, out[mask].T)
    idxs = np.argmax(sims, axis=0)
    return np.mean(np.all(vocab.vectors[idxs] == tgt[mask], axis=1))

pre_bind_acc = bind_accuracy(pre_bind_out, bind_vocab, test_bind_targets, n_present)
print(f"Binding retrieval accuracy (no training): {pre_bind_acc * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Train binding network with NaN-aware MSE
# ─────────────────────────────────────────────────────────────────────────────
print("Training binding network …")
n_bind_train = 2000

train_traces, train_cues, train_bind_targets, _ = get_binding_data(
    n_bind_train, n_pairs, dims, seed + 1, presentation_time, delay_time, dt
)

nan_mse = NanMSE()

with nengo_dl.Simulator(bind_net, minibatch_size=mini_sz, seed=seed) as sim:
    sim.compile(
        optimizer="rmsprop",
        loss={out_probe: nan_mse},
    )
    history_bind = sim.fit(
        x={trace_inp: train_traces, cue_inp: train_cues},
        y={out_probe: train_bind_targets},
        n_steps=n_steps,
        epochs=10,
    )
    sim.save_params("/tmp/spa_binding_params")
    print("Binding params saved to /tmp/spa_binding_params.npz")

    sim.run_steps(n_steps, data=bind_inputs)
    post_bind_out = sim.data[out_probe].copy()

post_bind_acc = bind_accuracy(post_bind_out, bind_vocab, test_bind_targets, n_present)
print(f"Binding retrieval accuracy (after training): {post_bind_acc * 100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Visualise binding retrieval (one example)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t_axis = np.arange(n_steps) * dt

sims_pre_bind  = spa.similarity(pre_bind_out[0],  bind_vocab)
sims_post_bind = spa.similarity(post_bind_out[0], bind_vocab)

axes[0].plot(t_axis, sims_pre_bind)
axes[0].axvline(presentation_time, color="k", linestyle="--", label="cue onset")
axes[0].set_ylabel("Cosine similarity")
axes[0].set_title("Binding retrieval (before training)")
axes[0].legend(loc="upper right")

axes[1].plot(t_axis, sims_post_bind)
axes[1].axvline(presentation_time, color="k", linestyle="--")
axes[1].set_ylabel("Cosine similarity")
axes[1].set_title("Binding retrieval (after training)")
axes[1].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("/tmp/spa_memory_binding.png", dpi=100)
print("Binding plot saved to /tmp/spa_memory_binding.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Training loss curves
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
ax1.plot(history_mem["loss"])
ax1.set_title("Simple-memory loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")

ax2.plot(history_bind["loss"])
ax2.set_title("Binding-memory loss (NaN-MSE)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("NaN-MSE")

plt.tight_layout()
plt.savefig("/tmp/spa_memory_loss.png", dpi=100)
print("Loss curves saved to /tmp/spa_memory_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
Results
───────
Simple-memory accuracy  – before training: {pre_mem_acc * 100:.1f}%
Simple-memory accuracy  – after training:  {post_mem_acc * 100:.1f}%

Binding retrieval       – before training: {pre_bind_acc * 100:.1f}%
Binding retrieval       – after training:  {post_bind_acc * 100:.1f}%

Key takeaways
─────────────
• A recurrent ensemble with synapse=tau sustains a semantic pointer over a
  delay period, mimicking working memory in neural circuits.
• Role⊛filler binding (circular convolution) compresses multiple associations
  into one vector; circular correlation retrieves a specific filler from a cue.
• NaN targets let us ignore the loss during the presentation phase and focus
  training on the retrieval phase.
• nengo-dl's sim.fit() tunes the ensemble weights to improve both tasks.
""")
