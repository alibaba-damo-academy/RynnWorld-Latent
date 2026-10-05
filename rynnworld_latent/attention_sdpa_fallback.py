"""Varlen-correct SDPA attention fallback for cosmos (no flash_attn needed).

Shared by scripts/train.py and rynnworld_latent/inference.py so every entry point
sees the identical attention backend set.
"""

from __future__ import annotations

import os

_PATCHED = False


def patch_sdpa_attention_backend() -> None:
    """Register a varlen-correct SDPA fallback into cosmos's attention frontend.

    Images without flash_attn/natten/cudnn-frontend leave cosmos with no
    compatible attention backend. This patch injects a ``"sdpa"`` backend that uses
    ``torch.nn.functional.scaled_dot_product_attention`` with a block-diagonal
    ``attn_mask`` constructed from ``cumulative_seqlen`` for the varlen case (no
    attention crosses packed-sequence boundaries). Also handles GQA and causal masking.

    Idempotent: both entry points call this, and re-applying would nest a second
    wrapper around the already-rebound ``choose_backend``.

    Disable with ``RYNNWORLD_SDPA_BACKEND=0``.
    """
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get("RYNNWORLD_SDPA_BACKEND") == "0":
        return
    try:
        import torch
        import torch.nn.functional as F
        import cosmos_framework.model.attention.frontend as _fe
    except Exception as e:  # noqa: BLE001
        print(f"[sdpa-fallback] skipping SDPA backend patch: {e}")
        return

    def sdpa_attention(
        query, key, value,
        is_causal=False, causal_type=None, scale=None,
        cumulative_seqlen_Q=None, cumulative_seqlen_KV=None,
        max_seqlen_Q=None, max_seqlen_KV=None,
        return_lse=False, backend_kwargs=None, deterministic=False,
    ):
        """SDPA backend. Varlen is handled segment-wise, never with a dense mask.

        A ``[S_Q, S_KV]`` mask is not an option here: sequence packing yields ~260k
        tokens (``max_samples_per_batch`` samples x ~8.2k tokens each), so the mask
        alone wanted 65 GiB. Slicing per packed sample instead keeps each SDPA call
        small, lets SDPA pick its memory-efficient/flash kernels (no materialized
        mask at all), and still guarantees no attention crosses a sample boundary.
        """
        B, S_Q, H_Q, D = query.shape
        _, S_KV, H_KV, _ = key.shape

        def _sdpa(q_bshd, k_bshd, v_bshd, causal):
            # [B,S,H,D] -> [B,H,S,D], run SDPA, transpose back.
            # GQA is handed to SDPA via enable_gqa rather than repeat_interleave:
            # expanding K/V from H_KV to H_Q heads materializes 2x the K/V tensors and
            # was enough to OOM the backward pass (71.75 of 79.18 GiB in use).
            q = q_bshd.permute(0, 2, 1, 3)
            k = k_bshd.permute(0, 2, 1, 3)
            v = v_bshd.permute(0, 2, 1, 3)
            if H_Q != H_KV:
                try:
                    o = F.scaled_dot_product_attention(
                        q, k, v, is_causal=causal, scale=scale, enable_gqa=True
                    )
                except TypeError:  # torch without enable_gqa: fall back to expansion
                    rep = H_Q // H_KV
                    o = F.scaled_dot_product_attention(
                        q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1),
                        is_causal=causal, scale=scale,
                    )
            else:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
            return o.permute(0, 2, 1, 3)

        if cumulative_seqlen_Q is None:
            out = _sdpa(query, key, value, is_causal)
        else:
            cq = cumulative_seqlen_Q.tolist()
            ckv = cumulative_seqlen_KV.tolist()
            chunks = []
            for s in range(len(cq) - 1):
                q_s, q_e = cq[s], cq[s + 1]
                k_s, k_e = ckv[s], ckv[s + 1]
                if q_e <= q_s:
                    continue
                chunks.append(
                    _sdpa(query[:, q_s:q_e], key[:, k_s:k_e], value[:, k_s:k_e], is_causal)
                )
            out = torch.cat(chunks, dim=1) if chunks else query.new_zeros(B, S_Q, H_Q, D)

        if return_lse:
            # SDPA does not expose logsumexp; callers here only use it for merging,
            # which this backend never triggers.
            return out, torch.zeros(B, S_Q, H_Q, 1, device=out.device, dtype=out.dtype)
        return out

    # Inject into BACKEND_MAP
    _fe.BACKEND_MAP["sdpa"] = sdpa_attention

    # Patch choose_backend to return "sdpa" as last resort when all others fail.
    from cosmos_framework.model.attention.backends import choose_backend as _orig_choose

    def _choose_with_sdpa_fallback(**kwargs):
        result = _orig_choose(**kwargs)
        if result is None and kwargs.get("backend") is None:
            return "sdpa"
        return result

    # Replace the choose_backend used by the frontend
    import cosmos_framework.model.attention.backends as _be
    _be.choose_backend = _choose_with_sdpa_fallback
    _fe.choose_backend = _choose_with_sdpa_fallback
    _PATCHED = True
    print("[sdpa-fallback] registered SDPA varlen attention backend (fallback for missing flash_attn)")
