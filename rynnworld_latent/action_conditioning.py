"""Action conditioning for the latent-action world model (final "film" recipe).

Cosmos3-Edge's MoT is a VLM-style transformer with NO AdaLN: every condition is a
token in the joint self-attention (or added onto tokens). The final architecture
("film") strengthens action injection with three complementary mechanisms, all
implemented as monkey-patches so the upstream cosmos source is never edited:

* **Two-tower action encoder** (``RYNNWORLD_ACTION_TOWER_SPLIT=1``)
    The 608-dim action is ``[k_tokens 512 | z 64 | camera 32]``. The single
    ``action2llm`` encoder is replaced by separate ``DomainAwareLinear`` towers for the
    non-camera (576-dim) and camera (32-dim) streams (summed) plus a shared residual
    MLP, so each modality gets its own nonlinear transform. ``.fc``/``.bias`` delegate
    to the non-camera tower so the stock ``init_weights`` keeps working.

* **Frame-aligned injection** (``RYNNWORLD_ACTION_FRAME_INJECT=1``)
    Action step j spans rgb frames [4j, 4j+4], i.e. exactly VAE latent frame j+1. This
    switch adds ``gate * action_token_j`` onto every vision token of latent frame j+1
    (gate: per-channel, zero-init -> exact identity at step 0, ControlNet-style). The
    gate lives at ``action2llm.frame_gate`` so the existing ``keys_to_skip_loading`` /
    lr-multiplier substring rules cover it.

* **Per-frame FiLM** (``RYNNWORLD_ACTION_FILM=1``)
    Each action token is mapped to (gamma, beta) over the hidden channels and applied
    multiplicatively to its frame's vision tokens:
    ``v *= (1 + gamma) + beta``. Zero-init -> identity at step 0.

Supporting knobs (also part of the final recipe):

* ``RYNNWORLD_ACTION_CFG_DROPOUT=<p>`` — action classifier-free-guidance dropout: with
  probability ``p`` a step's action tokens are zeroed during training, teaching an
  unconditional-action branch. At inference the same branch is selected via
  :data:`FORCE_UNCOND_ACTION` (see ``rynnworld_latent.inference``), giving pure action guidance.
* ``RYNNWORLD_ACTION_CFG_DROPOUT_HAND=<p>`` / ``RYNNWORLD_ACTION_CFG_DROPOUT_CAM=<p>`` —
  independent per-stream dropout (zero only the hand slice / only the camera slice),
  forcing the model to read each stream on its own and enabling per-stream guidance.
* ``RYNNWORLD_COND_FORCE=<p>`` / ``RYNNWORLD_COND_FORCE_FLOOR=<f>`` — condition-forcing, see
  :func:`register_cond_force_patches`.
* text-free is a pure config change (see ``experiment_config.py``): empty the caption
  and disable every metadata text augmentor so the only conditioning left is the action.

This module is imported by BOTH ``scripts/train.py`` (training) and
``rynnworld_latent/inference.py`` (inference) so the structure/weights line up. NOTE: eval of
a checkpoint trained with these switches must set the same env vars, otherwise the
extra parameters (``frame_gate`` / ``frame_film``) are missing from the model structure.
"""
from __future__ import annotations

import os

import torch
from torch import nn

# Global switch read by the action encoder: when True the encoder emits the
# unconditional (zero) action embedding. Set around the uncond velocity call at
# inference to realise action-CFG. Training never sets this (dropout is separate).
FORCE_UNCOND_ACTION = False

ACTION_CFG_DROPOUT = float(os.environ.get("RYNNWORLD_ACTION_CFG_DROPOUT", "0.0") or 0.0)
ACTION_CFG_DROPOUT_HAND = float(os.environ.get("RYNNWORLD_ACTION_CFG_DROPOUT_HAND", "0.0") or 0.0)
ACTION_CFG_DROPOUT_CAM = float(os.environ.get("RYNNWORLD_ACTION_CFG_DROPOUT_CAM", "0.0") or 0.0)
FRAME_INJECT = os.environ.get("RYNNWORLD_ACTION_FRAME_INJECT", "0").strip() == "1"
FRAME_FILM = os.environ.get("RYNNWORLD_ACTION_FILM", "0").strip() == "1"
TOWER_SPLIT = os.environ.get("RYNNWORLD_ACTION_TOWER_SPLIT", "0").strip() == "1"
# Non-camera stream size = everything before the trailing 32-dim camera slice, so
# for the 608-dim ktoken_zcam action it is 576 (k_tokens 512 + z 64). The camera
# tower is action_dim - HAND_DIM (= 32). See rynnworld_latent.dataset for the layout.
HAND_DIM = int(os.environ.get("RYNNWORLD_ACTION_HAND_DIM", "576"))


def _uncond_active() -> bool:
    return FORCE_UNCOND_ACTION


def _apply_action_cfg_dropout(x: torch.Tensor, training: bool, p_joint: float) -> torch.Tensor:
    """Uncond / CFG-dropout zeroing for the encoder wrapper.

    Order: inference uncond -> joint drop (full uncond anchor) -> independent
    per-stream drops (hand-only / cam-only null combinations for disentanglement).
    """
    if _uncond_active():
        return torch.zeros_like(x)
    if not training:
        return x
    if p_joint > 0.0 and torch.rand(1, device=x.device).item() < p_joint:
        return torch.zeros_like(x)
    if ACTION_CFG_DROPOUT_HAND > 0.0 and torch.rand(1, device=x.device).item() < ACTION_CFG_DROPOUT_HAND:
        x = torch.cat([torch.zeros_like(x[..., :HAND_DIM]), x[..., HAND_DIM:]], dim=-1)
    if ACTION_CFG_DROPOUT_CAM > 0.0 and torch.rand(1, device=x.device).item() < ACTION_CFG_DROPOUT_CAM:
        x = torch.cat([x[..., :HAND_DIM], torch.zeros_like(x[..., HAND_DIM:])], dim=-1)
    return x


class TwoTowerActionEncoder(nn.Module):
    """True two-tower action encoder: separate DomainAwareLinear for the hand
    (``x[..., :HAND_DIM]``) and camera (``x[..., HAND_DIM:]``) streams, summed, plus a
    shared residual MLP. Unlike splitting a plain Linear (a no-op), the per-stream
    residual MLPs give each modality its own nonlinear transform.

    ``.fc`` / ``.bias`` delegate to the hand tower so the stock ``init_weights``
    keeps working unchanged; the camera tower is always freshly (re)initialised by
    our patch.
    """

    def __init__(self, hand: nn.Module, cam: nn.Module, hidden: int, cfg_dropout: float):
        super().__init__()
        self.hand = hand
        self.cam = cam
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cfg_dropout = float(cfg_dropout)

    @property
    def fc(self):
        return self.hand.fc

    @property
    def bias(self):
        return self.hand.bias

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        x = _apply_action_cfg_dropout(x, self.training, self.cfg_dropout)
        h = self.hand(x[..., :HAND_DIM], domain_id) + self.cam(x[..., HAND_DIM:], domain_id)
        return h + self.mlp(h)


def _assign_full(param, full_cpu: torch.Tensor) -> None:
    """DTensor-safe write of a full CPU tensor into a (possibly FSDP-sharded) param."""
    tgt = param.data
    full = full_cpu.to(device=tgt.device, dtype=tgt.dtype)
    with torch.no_grad():
        try:
            from torch.distributed.tensor import DTensor, distribute_tensor

            if isinstance(tgt, DTensor):
                tgt.copy_(distribute_tensor(full, tgt.device_mesh, tgt.placements))
                return
        except Exception:  # noqa: BLE001 - fall back to local copy
            pass
        tgt.copy_(full)


def register_action_conditioning_patches() -> None:
    """Register the tower-split / CFG-dropout / frame-injection / FiLM patches on
    ``Cosmos3VFMNetwork``."""
    # Two knobs are structurally dependent on another switch. Letting them through
    # alone still prints "patches ON" while the model behaves as if they were unset,
    # which wastes a full run and — for CFG dropout — silently disables action
    # guidance at inference (the FORCE_UNCOND_ACTION branch becomes identical to the
    # conditional one).
    if (
        ACTION_CFG_DROPOUT > 0.0
        or ACTION_CFG_DROPOUT_HAND > 0.0
        or ACTION_CFG_DROPOUT_CAM > 0.0
    ) and not TOWER_SPLIT:
        raise RuntimeError(
            "RYNNWORLD_ACTION_CFG_DROPOUT/_HAND/_CAM require RYNNWORLD_ACTION_TOWER_SPLIT=1: "
            "the dropout and the FORCE_UNCOND_ACTION branch live in "
            "TwoTowerActionEncoder.forward, so with the stock single encoder they are "
            "no-ops and action guidance would do nothing."
        )
    if FRAME_FILM and not FRAME_INJECT:
        raise RuntimeError(
            "RYNNWORLD_ACTION_FILM=1 requires RYNNWORLD_ACTION_FRAME_INJECT=1: frame_film is "
            "created and applied inside the frame-aligned injection patch."
        )

    split_dropout = ACTION_CFG_DROPOUT_HAND > 0.0 or ACTION_CFG_DROPOUT_CAM > 0.0
    if (
        ACTION_CFG_DROPOUT <= 0.0
        and not split_dropout
        and not FRAME_INJECT
        and not TOWER_SPLIT
    ):
        # Nothing structural to add. (Pure text-free is config-only.)
        return

    try:
        from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
        from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear
    except Exception as e:  # noqa: BLE001
        print(f"[action_conditioning] skip (import failed: {e})")
        return

    if getattr(Cosmos3VFMNetwork, "_rynnworld_action_cond_patched", False):
        return
    Cosmos3VFMNetwork._rynnworld_action_cond_patched = True

    orig_init = Cosmos3VFMNetwork.__init__

    def _init_with_action_patches(self, language_model, config):
        orig_init(self, language_model, config)
        if not getattr(config, "action_gen", False):
            return
        if TOWER_SPLIT:
            # Replace the single 608-dim encoder with hand/cam towers + shared MLP.
            nd = self.num_embodiment_domains
            h = self.hidden_size
            hand = DomainAwareLinear(HAND_DIM, h, nd)
            cam = DomainAwareLinear(self.action_dim - HAND_DIM, h, nd)
            self.action2llm = TwoTowerActionEncoder(hand, cam, h, ACTION_CFG_DROPOUT)
        if FRAME_INJECT:
            # Per-channel gate, zero-init in init_weights -> exact identity at step 0.
            # Registered under action2llm so the "action2llm" substring rules
            # (keys_to_skip_loading on base load, lr multiplier) cover it.
            self.action2llm.frame_gate = nn.Parameter(torch.zeros(self.hidden_size))
            if FRAME_FILM:
                # Per-frame FiLM: action token -> (gamma, beta) over hidden channels,
                # applied multiplicatively to the controlled frame's vision tokens.
                # Zero-init -> gamma=beta=0 -> (1+0)*v+0 = v (identity at step 0).
                self.action2llm.frame_film = nn.Linear(self.hidden_size, 2 * self.hidden_size)

    Cosmos3VFMNetwork.__init__ = _init_with_action_patches

    orig_iw = Cosmos3VFMNetwork.init_weights

    def _init_weights_action_patches(self, buffer_device=None):
        orig_iw(self, buffer_device)
        a2l = getattr(self, "action2llm", None)
        h = self.hidden_size
        if isinstance(a2l, TwoTowerActionEncoder):
            # The hand tower is already zeroed: orig_iw is scripts/train.py's
            # action zero-init patch (registered first), which reached it through
            # the .fc delegation. Only the camera tower and the shared MLP are
            # ours to initialise.
            nd, h = self.num_embodiment_domains, self.hidden_size
            a_cam = self.action_dim - HAND_DIM
            _assign_full(a2l.cam.fc.weight, torch.zeros(nd, a_cam * h))
            _assign_full(a2l.cam.bias.weight, torch.zeros(nd, h))
            w0 = torch.empty(h, h)
            torch.nn.init.xavier_uniform_(w0)
            _assign_full(a2l.mlp[0].weight, w0)
            _assign_full(a2l.mlp[0].bias, torch.zeros(h))
            _assign_full(a2l.mlp[2].weight, torch.zeros(h, h))
            _assign_full(a2l.mlp[2].bias, torch.zeros(h))
        gate = getattr(a2l, "frame_gate", None)
        if gate is not None:
            _assign_full(gate, torch.zeros(h))
        film = getattr(a2l, "frame_film", None)
        if film is not None:
            _assign_full(film.weight, torch.zeros(2 * h, h))
            _assign_full(film.bias, torch.zeros(2 * h))

    Cosmos3VFMNetwork.init_weights = _init_weights_action_patches

    if FRAME_INJECT:
        orig_enc = Cosmos3VFMNetwork._encode_action

        def _encode_action_frame_inject(self, packed_seq, packed_sequence, target_dtype):
            orig_enc(self, packed_seq, packed_sequence, target_dtype)
            action = packed_seq.action
            vision = packed_seq.vision
            if (
                action is None
                or action.tokens is None
                or vision is None
                or not vision.token_shapes
                # dense action list must align 1:1 with the vision items; skip otherwise
                # (e.g. batches mixing action-less samples or multi-item image editing)
                or len(action.token_shapes) != len(vision.token_shapes)
            ):
                return
            gate = getattr(self.action2llm, "frame_gate", None)
            if gate is None:
                return
            # Read back the projected action tokens the original call just scattered in.
            act_tok = packed_sequence[action.sequence_indexes]  # [N_act, hidden]
            film = getattr(self.action2llm, "frame_film", None)
            gamma = beta = None
            if film is not None:
                gb = film(act_tok)  # [N_act, 2h]
                gamma, beta = gb.chunk(2, dim=-1)  # each [N_act, h]
            vis_idx = vision.sequence_indexes
            off_a = 0
            off_v = 0
            for shp_a, shp_v in zip(action.token_shapes, vision.token_shapes):
                t_a = shp_a[0]
                t_v, hh, ww = shp_v
                n_v = t_v * hh * ww
                # action step j spans rgb frames [4j,4j+4] -> exactly latent frame j+1
                if t_a == t_v - 1:
                    tgt = vis_idx[off_v + hh * ww : off_v + n_v]  # frames 1..t_v-1
                    if gamma is not None:
                        g_seg = gamma[off_a : off_a + t_a]  # [t_a, h]
                        b_seg = beta[off_a : off_a + t_a]  # [t_a, h]
                        g_rep = g_seg.repeat_interleave(hh * ww, dim=0)
                        b_rep = b_seg.repeat_interleave(hh * ww, dim=0)
                        packed_sequence[tgt] = (
                            packed_sequence[tgt] * (1.0 + g_rep.to(packed_sequence.dtype))
                            + b_rep.to(packed_sequence.dtype)
                        )
                    rep = act_tok[off_a : off_a + t_a].repeat_interleave(hh * ww, dim=0)
                    packed_sequence.index_add_(0, tgt, (gate * rep).to(packed_sequence.dtype))
                off_a += t_a
                off_v += n_v

        Cosmos3VFMNetwork._encode_action = _encode_action_frame_inject

    print(
        "[action_conditioning] patches ON: "
        f"cfg_dropout={ACTION_CFG_DROPOUT}, "
        f"dropout_hand={ACTION_CFG_DROPOUT_HAND}, dropout_cam={ACTION_CFG_DROPOUT_CAM}, "
        f"frame_inject={FRAME_INJECT}, film={FRAME_FILM}, tower_split={TOWER_SPLIT}"
    )


def patch_inference_action_guidance() -> None:
    """Make classifier-free guidance act on the ACTION (rynnworld_latent.inference only).

    Rewrites ``OmniMoTModel._run_classifier_free_guidance`` so the conditional branch
    uses the real action and the unconditional branch sets :data:`FORCE_UNCOND_ACTION`
    (zero action). Combined with text-free conditioning, the two branches differ only in
    the action, giving pure action guidance. Single-GPU path only (CFG-parallel disabled).
    """
    try:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    except Exception as e:  # noqa: BLE001
        print(f"[action_conditioning] skip guidance patch (import failed: {e})")
        return

    if getattr(OmniMoTModel, "_rynnworld_action_guidance_patched", False):
        return
    OmniMoTModel._rynnworld_action_guidance_patched = True

    import rynnworld_latent.action_conditioning as _ac

    def _run_cfg_action(self, cond_tokens, uncond_tokens, skip_text_tokens_for_cfg, single_velocity_fn):
        cond_v = single_velocity_fn(cond_tokens, False)
        _ac.FORCE_UNCOND_ACTION = True
        try:
            uncond_v = single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg)
        finally:
            _ac.FORCE_UNCOND_ACTION = False
        return cond_v, uncond_v

    OmniMoTModel._run_classifier_free_guidance = _run_cfg_action
    print("[action_conditioning] inference action-guidance patched (uncond branch zeroes action)")


def register_cond_force_patches() -> None:
    """Condition-forcing / pure-noise training to fight static collapse.

    With probability ``RYNNWORLD_COND_FORCE`` per sample, the sampled vision noise level
    sigma is raised to at least ``RYNNWORLD_COND_FORCE_FLOOR`` (near-pure-noise), so those
    steps cannot take the easy low-noise "copy frame 0" path and must instead generate
    the future from the conditioning frame + action. The paired timestep is recomputed
    from the floored sigma (``timestep = sigma * max_timestep``), so the denoiser's
    timestep embedding always matches the actual noise level.

    Off unless ``RYNNWORLD_COND_FORCE`` > 0. Training-only (guarded by ``self.training``).
    """
    prob = float(os.environ.get("RYNNWORLD_COND_FORCE", "0.0") or 0.0)
    if prob <= 0.0:
        return
    floor = float(os.environ.get("RYNNWORLD_COND_FORCE_FLOOR", "0.75"))

    try:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    except Exception as e:  # noqa: BLE001
        print(f"[action_conditioning] skip cond-force patch (import failed: {e})")
        return

    if getattr(OmniMoTModel, "_rynnworld_cond_force_patched", False):
        return
    OmniMoTModel._rynnworld_cond_force_patched = True

    orig = OmniMoTModel._get_train_noise_level_vision

    def _noise_level_with_cond_force(self, *args, **kwargs):
        timesteps, sigmas = orig(self, *args, **kwargs)
        if not self.training:
            return timesteps, sigmas
        # per-sample Bernoulli: force this sample to high noise?
        force = (torch.rand(sigmas.shape[0], 1, device=sigmas.device) < prob).to(sigmas.dtype)
        floored = torch.clamp(sigmas, min=floor)
        new_sigmas = force * floored + (1.0 - force) * sigmas
        # timestep = sigma * max_timestep; recover the (constant) scale from the originals
        scale = timesteps / sigmas.clamp_min(1e-6)
        new_timesteps = new_sigmas * scale
        return new_timesteps, new_sigmas

    OmniMoTModel._get_train_noise_level_vision = _noise_level_with_cond_force
    print(f"[action_conditioning] condition-forcing ON (p={prob}, sigma floor={floor})")
