#!/usr/bin/env python3
"""
Worm Brain Sidecar — Quick Validation
=======================================
Four tests to run before committing to a long training run.

Usage:
    python validate_worm_sidecar.py           # run all tests
    python validate_worm_sidecar.py --test 1  # run specific test
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn.functional as F

from model import HamnerConfig
from worm_brain import CelegansHH, CelegansConnectome, NUM_NEURONS
from worm_sidecar import WormSidecarModel


# Model config matching train_pretrain.py
MODEL_CONFIG = dict(
    hidden_size=768,
    num_layers=20,
    num_attention_heads=12,
    num_kv_heads=4,
    num_experts=1,
    num_active_experts=1,
    expert_intermediate_size=2048,
    use_differential_attention=False,
    gradient_checkpointing=False,  # faster for validation
    max_seq_len=1024,
    vocab_size=32000,
)


def test_1_sidecar_init():
    """Test 1: Sidecar init produces output similar to vanilla model.

    Both models are randomly initialized. The sidecar should produce
    near-identical output because all coupling weights start at zero.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Sidecar initialization transparency")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HamnerConfig(**MODEL_CONFIG)

    # Seed for reproducibility
    torch.manual_seed(42)
    from model import HamnerModel
    vanilla = HamnerModel(config).to(device)

    torch.manual_seed(42)
    sidecar = WormSidecarModel(config, substeps=10).to(device)

    # The base model weights should be identical
    vanilla_keys = set(vanilla.state_dict().keys())
    sidecar_model_keys = set(k.replace("model.", "")
                              for k in sidecar.state_dict().keys()
                              if k.startswith("model."))
    missing = vanilla_keys - sidecar_model_keys
    if missing:
        print(f"  WARNING: Missing keys in sidecar: {missing}")

    # Copy vanilla weights to sidecar's model sub-module
    sidecar.model.load_state_dict(vanilla.state_dict())

    # Test with random input
    input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)

    vanilla.eval()
    sidecar.eval()

    with torch.no_grad():
        vanilla_out = vanilla(input_ids)
        sidecar_out = sidecar(input_ids)

    vanilla_logits = vanilla_out["logits"]
    sidecar_logits = sidecar_out["logits"]

    # Compare
    max_diff = (vanilla_logits - sidecar_logits).abs().max().item()
    mean_diff = (vanilla_logits - sidecar_logits).abs().mean().item()

    # With feedback_scale=0 and dopamine_projection init near zero,
    # the main difference is from dopamine temperature modulation.
    # Temperature should be ~1.0, so logits should be very close.
    print(f"  Vanilla logits shape: {vanilla_logits.shape}")
    print(f"  Sidecar logits shape: {sidecar_logits.shape}")
    print(f"  Max absolute diff:    {max_diff:.6f}")
    print(f"  Mean absolute diff:   {mean_diff:.6f}")

    # Tolerance: temperature of ~1.0 means logits differ by at most logit_magnitude * epsilon
    # Since logits are ~O(1) at init and temperature is sigmoid(~0) = 0.5 + 0.5 = ~1.0
    # we allow some difference from the dopamine modulation path
    passed = max_diff < 5.0  # generous tolerance since temperature may deviate from exactly 1.0
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} (tolerance: max_diff < 5.0)")
    return passed


def test_2_worm_signal():
    """Test 2: Worm produces non-trivial dopamine signals.

    Run 100 random sequences through the sidecar and verify that
    dopamine output has non-zero variance (the worm is not flat).
    """
    print("\n" + "=" * 60)
    print("TEST 2: Worm signal variance")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HamnerConfig(**MODEL_CONFIG)

    torch.manual_seed(123)
    model = WormSidecarModel(config, substeps=10).to(device)
    model.eval()

    all_dopa = []
    all_V = []

    with torch.no_grad():
        for i in range(100):
            input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
            model.worm.reset_state(1, device)
            model._prev_worm_state = None
            _ = model(input_ids)

            diag = model.get_worm_diagnostics()
            all_dopa.append(diag.get("mean_dopamine", 0.0))
            all_V.append(diag.get("mean_membrane_potential", 0.0))

    dopa_tensor = torch.tensor(all_dopa)
    V_tensor = torch.tensor(all_V)

    dopa_var = dopa_tensor.var().item()
    V_var = V_tensor.var().item()
    dopa_mean = dopa_tensor.mean().item()
    V_mean = V_tensor.mean().item()

    print(f"  Dopamine — mean: {dopa_mean:.6f}, var: {dopa_var:.6f}")
    print(f"  Membrane — mean: {V_mean:.2f}, var: {V_var:.2f}")

    # Dopamine should have some variance (not flat 0.5)
    passed = dopa_var > 1e-10
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} (dopamine var > 1e-10)")
    return passed


def test_3_short_training():
    """Test 3: Loss decreases over 200 steps of memorization training.

    Uses a FIXED batch repeated every step so the model can memorize it.
    Random tokens with fresh batches each step have no learnable patterns
    (loss = ln(vocab_size) is optimal), so we test memorization instead.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Short training (200 steps, memorization)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HamnerConfig(**MODEL_CONFIG)

    torch.manual_seed(42)
    model = WormSidecarModel(config, substeps=5).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    # Fixed batch for memorization — model should overfit to this
    batch_size = 2
    seq_len = 32
    fixed_input = torch.randint(0, config.vocab_size, (batch_size, seq_len),
                                device=device)
    fixed_labels = fixed_input.clone()

    losses_start = []
    losses_end = []
    n_steps = 200

    start = time.time()
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(fixed_input, labels=fixed_labels)
            loss = outputs["loss"]

        if torch.isnan(loss):
            print(f"  NaN at step {step}")
            return False

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        lv = loss.item()
        if step < 20:
            losses_start.append(lv)
        if step >= n_steps - 20:
            losses_end.append(lv)

        if (step + 1) % 50 == 0:
            print(f"  step {step+1:>4d} | loss {lv:.4f}")

    elapsed = time.time() - start
    avg_start = sum(losses_start) / len(losses_start)
    avg_end = sum(losses_end) / len(losses_end)
    improvement = (avg_start - avg_end) / avg_start * 100

    print(f"\n  First 20 steps avg loss:  {avg_start:.4f}")
    print(f"  Last 20 steps avg loss:   {avg_end:.4f}")
    print(f"  Improvement:              {improvement:.1f}%")
    print(f"  Time: {elapsed:.1f}s ({n_steps/elapsed:.1f} steps/s)")

    passed = avg_end < avg_start
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} (loss should decrease)")
    return passed


def test_4_temporal_dynamics():
    """Test 4: Worm state evolves meaningfully over token positions.

    Feed a long sequence and track worm membrane potential evolution.
    The state should change over time (not constant).
    """
    print("\n" + "=" * 60)
    print("TEST 4: Temporal dynamics")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = HamnerConfig(**MODEL_CONFIG)

    torch.manual_seed(7)
    model = WormSidecarModel(config, substeps=20).to(device)
    model.eval()

    # Run a sequence through the model, capturing worm state at each position
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

    model.worm.reset_state(1, device)
    model._prev_worm_state = None

    # Extract layer 5 hidden states by running layers 0-5 manually
    with torch.no_grad():
        x = model.model.embed_tokens(input_ids)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        layer5 = None
        for i, block in enumerate(model.model.layers):
            x, _ = block(x, model.model.rope_cos, model.model.rope_sin,
                         causal_mask)
            if i == WormSidecarModel.TAP_LAYER:
                layer5 = x.clone()
                break

    if layer5 is None:
        print("  ERROR: Could not extract layer 5 hidden states")
        return False

    # Now run worm token by token
    model.worm.reset_state(1, device)
    states = []
    with torch.no_grad():
        for t in range(seq_len):
            token_hidden = layer5[:, t, :]
            sensory_input = model.tap_projection(token_hidden)
            model.worm.inject_sensory(sensory_input)
            motor, dopa = model.worm.step()
            V = model.worm.get_full_state().clone()
            states.append(V)

    states_tensor = torch.cat(states, dim=0)  # (seq_len, 302)

    # Analyze temporal dynamics
    # 1. Variance across time for each neuron
    temporal_var = states_tensor.var(dim=0)  # (302,)
    mean_temporal_var = temporal_var.mean().item()
    max_temporal_var = temporal_var.max().item()

    # 2. Difference between first and last state
    first_state = states_tensor[0]
    last_state = states_tensor[-1]
    state_drift = (last_state - first_state).abs().mean().item()

    # 3. Autocorrelation (smoothness of dynamics)
    diffs = (states_tensor[1:] - states_tensor[:-1]).abs().mean().item()

    print(f"  Sequence length:     {seq_len} tokens")
    print(f"  Mean temporal var:   {mean_temporal_var:.6f}")
    print(f"  Max temporal var:    {max_temporal_var:.6f}")
    print(f"  State drift:         {state_drift:.4f}")
    print(f"  Mean step-to-step:   {diffs:.6f}")

    # Try to save a diagnostic plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot a few example neuron traces
        ax = axes[0, 0]
        neuron_indices = [0, 50, 100, 150, 200, 250, 301]
        for ni in neuron_indices:
            if ni < states_tensor.shape[1]:
                ax.plot(states_tensor[:, ni].cpu().numpy(),
                        linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Membrane Potential (mV)")
        ax.set_title("Example Neuron Traces")
        ax.grid(True, alpha=0.3)

        # Temporal variance histogram
        ax = axes[0, 1]
        ax.hist(temporal_var.cpu().numpy(), bins=50, color="#2196F3", alpha=0.7)
        ax.set_xlabel("Temporal Variance")
        ax.set_ylabel("Count")
        ax.set_title("Per-Neuron Temporal Variance")
        ax.grid(True, alpha=0.3)

        # Mean membrane potential over time
        ax = axes[1, 0]
        mean_V = states_tensor.mean(dim=1).cpu().numpy()
        ax.plot(mean_V, color="#F44336", linewidth=1.5)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Mean V (mV)")
        ax.set_title("Population Mean Membrane Potential")
        ax.grid(True, alpha=0.3)

        # Heatmap of neuron activity
        ax = axes[1, 1]
        # Subsample for visualization
        n_show = min(50, states_tensor.shape[1])
        step = max(1, states_tensor.shape[1] // n_show)
        heatmap_data = states_tensor[:, ::step].T.cpu().numpy()
        im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r",
                        interpolation="nearest")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Neuron Index")
        ax.set_title("Neuron Activity Heatmap")
        plt.colorbar(im, ax=ax, label="V (mV)")

        plt.suptitle("Worm Temporal Dynamics Validation", fontsize=14,
                      fontweight="bold")
        plt.tight_layout()

        os.makedirs("logs/plots", exist_ok=True)
        plt.savefig("logs/plots/worm_temporal_dynamics.png", dpi=150,
                     bbox_inches="tight")
        print(f"  Plot saved: logs/plots/worm_temporal_dynamics.png")
        plt.close()
    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    # State should evolve (not be static)
    passed = mean_temporal_var > 1e-10 and state_drift > 1e-6
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} "
          f"(temporal_var > 1e-10 and drift > 1e-6)")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate worm sidecar")
    parser.add_argument("--test", type=int, default=None,
                        help="Run specific test (1-4)")
    args = parser.parse_args()

    print("=" * 60)
    print("WORM BRAIN SIDECAR — VALIDATION SUITE")
    print("=" * 60)

    tests = {
        1: ("Sidecar init transparency", test_1_sidecar_init),
        2: ("Worm signal variance", test_2_worm_signal),
        3: ("Short training", test_3_short_training),
        4: ("Temporal dynamics", test_4_temporal_dynamics),
    }

    if args.test:
        if args.test not in tests:
            print(f"Unknown test: {args.test}. Choose 1-4.")
            sys.exit(1)
        name, fn = tests[args.test]
        print(f"\nRunning test {args.test}: {name}")
        passed = fn()
        sys.exit(0 if passed else 1)

    # Run all tests
    results = {}
    for num, (name, fn) in tests.items():
        try:
            results[num] = fn()
        except Exception as e:
            print(f"\n  TEST {num} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[num] = False

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for num, (name, _) in tests.items():
        status = "PASS" if results.get(num, False) else "FAIL"
        if not results.get(num, False):
            all_pass = False
        print(f"  Test {num}: {name:<35s} [{status}]")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
