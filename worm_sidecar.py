"""
Worm Brain Sidecar — Wrapper Module
====================================
Integrates a C. elegans neural simulation alongside the HamnerModel transformer.

Architecture:
    Token input → Embedding → [+feedback] → Layers 0-5 → tap → Worm simulation
                                             Layers 6-19 → final_norm → lm_head
                                                                        ↓
                                             logits → dopamine modulation → output

The worm sidecar adds ~279K parameters (<1MB VRAM) on top of the 164M transformer.

Uses forward hooks for layer 5 capture and feedback injection, allowing
the core transformer to be compiled as a single unit with graph breaks
only at the hook points.
"""

from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import HamnerModel, HamnerConfig
from worm_brain import CelegansHH, NUM_NEURONS


def _worm_process_tokens(worm, tap_projection, sensory_idx_t, strided_hidden):
    """Process strided token hidden states through the worm simulation.

    This function is designed to be torch.compile-friendly: no in-place ops,
    no module state mutation (state is threaded through explicitly).

    Args:
        worm: CelegansHH module (for parameters/buffers)
        tap_projection: nn.Linear(hidden_size, num_sensory)
        sensory_idx_t: (num_sensory,) index tensor
        strided_hidden: (bsz, n_tokens, hidden_size) fp32 tensor

    Returns:
        mean_dopa: (bsz, num_dopa) averaged dopamine across tokens
        final_state: (bsz, 302, 5) final worm state
    """
    bsz, n_tokens, _ = strided_hidden.shape
    state = worm._state
    dopamine_sum = torch.zeros(bsz, worm.num_dopa, device=strided_hidden.device)

    for t in range(n_tokens):
        token_hidden = strided_hidden[:, t, :]
        sensory_input = tap_projection(token_hidden)

        # Build input current (no in-place)
        I_input = torch.zeros(bsz, NUM_NEURONS, device=strided_hidden.device)
        I_input = I_input.scatter(
            1, sensory_idx_t.unsqueeze(0).expand(bsz, -1), sensory_input
        )

        # Integrate
        state = worm._integrate(state, I_input)

        # Accumulate dopamine
        V = state[:, :, 0]
        dopa = torch.sigmoid(V[:, worm.dopa_idx_t] / 20.0)
        dopamine_sum = dopamine_sum + dopa

    mean_dopa = dopamine_sum / n_tokens
    return mean_dopa, state


class WormSidecarModel(nn.Module):
    """Wraps HamnerModel with a C. elegans neural simulation sidecar.

    The worm simulation runs in parallel with the transformer:
    1. A tap projection extracts a signal from layer 5 hidden states
    2. This drives the 302-neuron worm simulation
    3. Dopamine output modulates the final logits temperature
    4. Full worm state feeds back into the embedding for the next batch

    The sidecar is designed to be near-transparent at initialization
    (all coupling weights start near zero), so it doesn't disrupt training.

    Uses forward hooks on embed_tokens (feedback injection) and layer 5
    (hidden state capture). These cause graph breaks when torch.compile
    is used, splitting the compiled model into segments.
    """

    TAP_LAYER = 5  # Which transformer layer to tap

    def __init__(self, config: HamnerConfig, substeps: int = 100,
                 worm_stride: int = 1):
        super().__init__()

        self.config = config
        self.worm_stride = worm_stride  # process every Nth token position
        hidden_size = config.hidden_size

        # Core transformer
        self.model = HamnerModel(config)

        # Worm simulation
        self.worm = CelegansHH(substeps=substeps)

        # --- Sidecar projection layers ---

        # Tap: project hidden state → sensory neuron space
        self.tap_projection = nn.Linear(hidden_size, self.worm.num_sensory, bias=False)
        nn.init.xavier_uniform_(self.tap_projection.weight, gain=0.1)

        # Dopamine: map [worm_dopa(8) + tap_bypass(num_sensory)] → scalar temperature
        # The tap bypass provides a differentiable gradient path (worm is non-diff).
        # Worm dopamine gives the biological signal; bypass gives gradients so
        # tap_projection and this layer can learn.
        dopa_input_size = self.worm.num_dopa + self.worm.num_sensory
        self.dopamine_projection = nn.Sequential(
            nn.Linear(dopa_input_size, 1, bias=True),
            nn.Sigmoid(),
        )
        # Init with small random weights so gradients flow through from step 1.
        # Small enough that sigmoid(w@x+b) ≈ sigmoid(0) = 0.5, so temp ≈ 1.0.
        nn.init.xavier_uniform_(self.dopamine_projection[0].weight, gain=0.01)
        nn.init.zeros_(self.dopamine_projection[0].bias)

        # Feedback: map full worm state → residual stream
        # Init with small random weights (NOT zeros — zero * zero = dead gradient)
        self.feedback_projection = nn.Linear(NUM_NEURONS, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.feedback_projection.weight, gain=0.01)

        # Learnable gating coefficient for feedback, constrained to [0.001, 0.051]
        # via sigmoid reparameterization so it can never reach zero.
        # Init raw param so sigmoid(-1.516) ≈ 0.18 → scale ≈ 0.01
        self._feedback_scale_raw = nn.Parameter(torch.tensor(-1.516))

        # State carried across batches
        self._prev_worm_state = None

        # Hook state (set before each forward, read by hooks)
        self._layer5_captured = None
        self._feedback_to_inject = None

        # Register hooks on the transformer submodules
        self._setup_hooks()

    @property
    def feedback_scale(self):
        """Constrained feedback scale in [0.001, 0.051]. Never zero."""
        return 0.001 + 0.05 * torch.sigmoid(self._feedback_scale_raw)

    def _setup_hooks(self):
        """Register forward hooks for layer 5 capture and feedback injection.

        These hooks cause graph breaks when torch.compile wraps self.model,
        splitting the compiled graph into segments. This is the intended
        behavior — the segments compile independently.
        """
        def tap_hook(module, input, output):
            x, aux_loss = output
            self._layer5_captured = x.detach()
            return output

        def feedback_hook(module, input, output):
            if self._feedback_to_inject is not None:
                return output + self._feedback_to_inject
            return output

        self.model.layers[self.TAP_LAYER].register_forward_hook(tap_hook)
        self.model.embed_tokens.register_forward_hook(feedback_hook)

    @contextmanager
    def uncompiled(self):
        """Temporarily swap to uncompiled model to avoid recompilation.

        Use for validation (train/eval mode switch triggers recompilation
        when gradient_checkpointing=True) and generation (varying seq_len
        triggers recompilation per shape).
        """
        compiled = self.model
        if hasattr(compiled, '_orig_mod'):
            self.model = compiled._orig_mod
        try:
            yield
        finally:
            self.model = compiled

    def forward(self, input_ids, labels=None, attention_mask=None):
        """Forward pass with worm sidecar processing.

        Uses hooks on embed_tokens and layer 5 to inject feedback and
        capture hidden states without modifying model.py.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Reset worm state for this batch if needed
        if self.worm._state is None or self.worm._state.shape[0] != bsz:
            self.worm.reset_state(bsz, device)
            self._prev_worm_state = None

        # --- Prepare feedback injection (hook reads this) ---
        if self._prev_worm_state is not None:
            feedback = self.feedback_scale * self.feedback_projection(
                self._prev_worm_state.to(device)
            )
            self._feedback_to_inject = feedback.unsqueeze(1)  # (bsz, 1, hidden)
        else:
            self._feedback_to_inject = None

        # Reset captured state
        self._layer5_captured = None

        # --- Run transformer (hooks capture layer 5 + inject feedback) ---
        # Don't pass labels — we compute loss ourselves after temperature modulation
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        total_aux_loss = outputs["aux_loss"]
        layer5_hidden = self._layer5_captured

        # --- Worm processing ---
        temperature = None
        if layer5_hidden is not None:
            layer5_fp32 = layer5_hidden.float()
            strided = layer5_fp32[:, ::self.worm_stride, :]

            # Differentiable tap bypass: compute tap signal WITH gradients.
            # This gives tap_projection and dopamine_projection a gradient path
            # that bypasses the non-differentiable worm simulation.
            # layer5_hidden is detached from the transformer (hook detaches it),
            # so gradients flow to tap_projection weights but NOT back into the
            # transformer — keeping the transformer isolated from worm influence.
            tap_mean = self.tap_projection(layer5_fp32.mean(dim=1))  # (bsz, num_sensory)

            # Non-differentiable worm biological dynamics
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
                mean_dopa, final_state = _worm_process_tokens(
                    self.worm, self.tap_projection,
                    self.worm.sensory_idx_t, strided,
                )

            # Update worm state
            self.worm._state = final_state.detach()

            # Compute temperature from worm dopamine + differentiable tap bypass.
            # Worm dopamine (8-dim, no grad) = biological signal
            # Tap bypass (86-dim, has grad) = differentiable gradient path
            dopa_input = torch.cat([mean_dopa, tap_mean], dim=-1)
            temperature = 0.5 + self.dopamine_projection(dopa_input)
            self._last_temperature = temperature.detach()

            # Store worm state for next-batch feedback
            self._prev_worm_state = final_state[:, :, 0].detach()

        # Apply temperature and compute loss.
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            del logits  # free ~2.3GB
            if temperature is not None:
                temp_expanded = temperature.unsqueeze(1).expand(bsz, seq_len - 1, 1).reshape(-1, 1)
                shift_logits = shift_logits / temp_expanded
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            loss = loss + total_aux_loss
            return {"loss": loss, "logits": None, "aux_loss": total_aux_loss}

        # Inference path
        if temperature is not None:
            logits = logits / temperature.unsqueeze(1)

        return {"loss": None, "logits": logits, "aux_loss": total_aux_loss}

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def count_sidecar_parameters(self):
        """Count parameters added by the sidecar (excluding base model)."""
        model_params = set(id(p) for p in self.model.parameters())
        sidecar_total = 0
        sidecar_trainable = 0
        for p in self.parameters():
            if id(p) not in model_params:
                sidecar_total += p.numel()
                if p.requires_grad:
                    sidecar_trainable += p.numel()
        return sidecar_total, sidecar_trainable

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8,
                 top_k=50, top_p=0.9, repetition_penalty=1.1, eos_token_id=2):
        """Autoregressive generation with worm sidecar active."""
        self.eval()
        bsz = input_ids.shape[0]

        # Reset worm for generation
        self.worm.reset_state(bsz, input_ids.device)
        self._prev_worm_state = None

        with self.uncompiled():
            for _ in range(max_new_tokens):
                idx_cond = input_ids[:, -self.config.max_seq_len:]
                outputs = self(idx_cond)
                logits = outputs["logits"][:, -1, :]

                if repetition_penalty != 1.0:
                    for tid in set(input_ids[0].tolist()):
                        logits[0, tid] /= repetition_penalty

                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p < 1.0:
                    sl, si = torch.sort(logits, descending=True)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    rm = cp > top_p
                    rm[..., 1:] = rm[..., :-1].clone()
                    rm[..., 0] = False
                    logits[rm.scatter(1, si, rm)] = float("-inf")

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == eos_token_id:
                    break

        return input_ids

    def load_base_model(self, checkpoint_path: str, device="cpu"):
        """Load a vanilla HamnerModel checkpoint into the model sub-module."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned, strict=True)
        return ckpt.get("config", {})

    def get_sidecar_state_dict(self):
        """Return only sidecar parameters (for separate saving/loading)."""
        prefix_set = {"tap_projection.", "dopamine_projection.",
                       "feedback_projection.", "_feedback_scale_raw", "worm."}
        result = {}
        for k, v in self.state_dict().items():
            if any(k.startswith(p) for p in prefix_set):
                result[k] = v
        return result

    def get_worm_diagnostics(self):
        """Return diagnostic info about the worm's current state."""
        if self.worm._state is None:
            return {}

        V = self.worm._state[:, :, 0]  # (batch, 302)
        da_level = self.worm.get_da_level()

        temp = getattr(self, '_last_temperature', None)
        return {
            "mean_membrane_potential": V.mean().item(),
            "membrane_potential_std": V.std().item(),
            "mean_dopamine": da_level.mean().item() if da_level is not None else 0.0,
            "dopamine_std": da_level.std(unbiased=False).item() if da_level is not None else 0.0,
            "feedback_scale": self.feedback_scale.item(),
            "worm_synaptic_weight_mean": self.worm.chem_weights.mean().item(),
            "worm_synaptic_weight_std": self.worm.chem_weights.std().item(),
            "temperature_mean": temp.mean().item() if temp is not None else 1.0,
            "temperature_std": temp.std().item() if temp is not None else 0.0,
        }
