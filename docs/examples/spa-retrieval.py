"""spa-retrieval.py
==================
Semantic Pointer Architecture (SPA) – Cued Retrieval via Circular Convolution.

This example mirrors the original spa-retrieval notebook.  It shows how to
train a Nengo SPA network to retrieve a filler semantic pointer when cued with
its associated role, using role⊛filler binding and circular-correlation
(inverse convolution) for retrieval.

Outline
-------
1. Generate structured vocabularies: each example stores ``n_pairs``
   role/filler pairs bound together into a single "trace" pointer.
2. Build a ``CircularConvolution`` network that computes
   ``output ≈ trace ⊛⁻¹ cue``.
3. Without training: evaluate retrieval accuracy (may be low for random
   weights).
4. Fine-tune the network with nengo-dl to maximise accuracy.
5. Evaluate and visualise the output similarity to the vocabulary.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import nengo
from nengo import spa
import nengo_dl

# ── reproducibility ───────────────────────────────────────────────────────────
seed = 0

# ── 1. Data generation ────────────────────────────────────────────────────────
def get_data(n_items, pairs_per_item, vec_d, vocab_seed):
    """Generate role/filler binding examples.

    Returns
    -------
    traces : (n_items, 1, vec_d)  – sum of bound role*filler pairs, normalised
    cues   : (n_items, 1, vec_d)  – the query role pointer
    targets: (n_items, 1, vec_d)  – the expected filler pointer
    vocab  : nengo.spa.Vocabulary
    """
    rng = np.random.RandomState(vocab_seed)
    vocab = spa.Vocabulary(dimensions=vec_d, rng=rng, max_similarity=1)

    traces  = np.zeros((n_items, 1, vec_d))
    cues    = np.zeros((n_items, 1, vec_d))
    targets = np.zeros((n_items, 1, vec_d))

    for n in range(n_items):
        role_names   = [f"ROLE_{n}_{i}"   for i in range(pairs_per_item)]
        filler_names = [f"FILLER_{n}_{i}" for i in range(pairs_per_item)]

        # trace = normalised sum of role_i * filler_i
        trace_key = f"TRACE_{n}"
        trace_ptr = vocab.parse(
            "+".join(f"{r} * {f}" for r, f in zip(role_names, filler_names))
        )
        trace_ptr.normalize()
        vocab.add(trace_key, trace_ptr)

        # choose a random role as the retrieval cue
        cue_idx = rng.randint(pairs_per_item)
        traces[n, 0, :]  = vocab[trace_key].v
        cues[n, 0, :]    = vocab[role_names[cue_idx]].v
        targets[n, 0, :] = vocab[filler_names[cue_idx]].v

    return traces, cues, targets, vocab


# ── 2. Network ────────────────────────────────────────────────────────────────
seed_net  = seed
dims      = 32
n_pairs   = 2
mini_size = 50

with nengo.Network(seed=seed_net) as net:
    net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    net.config[nengo.Connection].synapse   = None

    trace_inp = nengo.Node(np.zeros(dims))
    cue_inp   = nengo.Node(np.zeros(dims))

    # CircularConvolution with invert_b=True computes trace ⊛⁻¹ cue
    cconv = nengo.networks.CircularConvolution(5, dims, invert_b=True)
    nengo.Connection(trace_inp, cconv.input_a)
    nengo.Connection(cue_inp,   cconv.input_b)

    out_probe = nengo.Probe(cconv.output)

print(f"Network: CircularConvolution(n_neurons=5, dims={dims})")
print(f"Vocabulary: {dims}-D vectors, {n_pairs} role/filler pairs per item")


# ── 3. Accuracy helper ────────────────────────────────────────────────────────
def accuracy(output, vocab, targets, t_step=-1):
    """Fraction of outputs whose nearest neighbour matches the target."""
    out  = output[:, t_step, :]                    # (batch, dims)
    sims = np.dot(vocab.vectors, out.T)            # (n_vocab, batch)
    idxs = np.argmax(sims, axis=0)
    return np.mean(np.all(vocab.vectors[idxs] == targets[:, 0], axis=1))


# ── 4. Baseline (no training) ─────────────────────────────────────────────────
test_traces, test_cues, test_targets, test_vocab = get_data(
    mini_size, n_pairs, dims, vocab_seed=seed
)
test_inputs = {trace_inp: test_traces, cue_inp: test_cues}

with nengo_dl.Simulator(net, minibatch_size=mini_size, seed=seed) as sim:
    sim.run_steps(1, data=test_inputs)
    baseline_acc = accuracy(sim.data[out_probe], test_vocab, test_targets)
    baseline_out = sim.data[out_probe][0, 0].copy()

print(f"\nRetrieval accuracy (no training): {baseline_acc * 100:.1f}%")

# Visualise similarity to vocabulary for first example
plt.figure(figsize=(10, 4))
sims_baseline = np.dot(test_vocab.vectors, baseline_out)
bars = plt.bar(np.arange(len(test_vocab.vectors)), sims_baseline)
target_idx = np.where(
    np.all(test_vocab.vectors == test_targets[0, 0], axis=1)
)[0]
if len(target_idx):
    bars[target_idx[0]].set_color("red")
plt.ylim([-1, 1])
plt.xlabel("Vocabulary item index")
plt.ylabel("Cosine similarity")
plt.title("Output similarity to vocabulary (baseline, red = target)")
plt.tight_layout()
plt.show()


# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\nTraining …")
n_train  = 5000

train_traces, train_cues, train_targets, _ = get_data(
    n_train, n_pairs, dims, vocab_seed=seed + 1
)

with nengo_dl.Simulator(net, minibatch_size=mini_size, seed=seed) as sim:
    sim.compile(optimizer="rmsprop", loss={out_probe: "mse"})

    # The target for the probe is the filler vector at each timestep
    history = sim.fit(
        x={trace_inp: train_traces, cue_inp: train_cues},
        y={out_probe: train_targets},
        n_steps=1,
        epochs=20,
    )
    sim.save_params("/tmp/spa_retrieval_params")
    print("Params saved to /tmp/spa_retrieval_params.npz")

    # Evaluate on test set
    sim.run_steps(1, data=test_inputs)
    trained_acc = accuracy(sim.data[out_probe], test_vocab, test_targets)
    trained_out = sim.data[out_probe][0, 0].copy()

print(f"Retrieval accuracy (after training): {trained_acc * 100:.1f}%")


# ── 6. Visualise trained output ───────────────────────────────────────────────
plt.figure(figsize=(10, 4))
sims_trained = np.dot(test_vocab.vectors, trained_out)
bars = plt.bar(np.arange(len(test_vocab.vectors)), sims_trained)
if len(target_idx):
    bars[target_idx[0]].set_color("red")
plt.ylim([-1, 1])
plt.xlabel("Vocabulary item index")
plt.ylabel("Cosine similarity")
plt.title("Output similarity to vocabulary (trained, red = target)")
plt.tight_layout()
plt.show()


# ── 7. Training loss curve ────────────────────────────────────────────────────
plt.figure(figsize=(6, 3))
plt.plot(history["loss"])
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("SPA retrieval training loss")
plt.tight_layout()
plt.show()


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
Results
───────
Baseline accuracy (untrained):  {baseline_acc * 100:.1f}%
Trained accuracy:                {trained_acc * 100:.1f}%

Key takeaways
─────────────
• Semantic pointers use circular convolution (⊛) for binding:
    trace = normalise(role_1 ⊛ filler_1 + role_2 ⊛ filler_2 + …)
• Retrieval: output = trace ⊛⁻¹ cue  (circular correlation)
• A Nengo CircularConvolution network implements this in neural hardware.
• Training with nengo-dl's sim.fit() fine-tunes the internal ensemble
  weights to improve retrieval accuracy on new role/filler pairs.
• The red bar in the similarity plots should be the tallest after training.
""")
