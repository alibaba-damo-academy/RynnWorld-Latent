# Third-party licenses

RynnWorld-Latent (Apache-2.0, see [LICENSE](LICENSE)) bundles the components
below. Each component's full license text and notices ship in its own
directory; this file is a summary index. **No model weights are redistributed**
— obtain them with [`scripts/setup/download_weights.sh`](scripts/setup/download_weights.sh).

| Component | Path | License | Copyright |
|---|---|---|---|
| cosmos-framework | `third_party/cosmos-framework/` | NVIDIA OpenMDW-1.1 | NVIDIA Corp. & Affiliates |
| cosmos tokenizers (text) | `third_party/cosmos_tokenizers/` | NVIDIA OpenMDW-1.1 | NVIDIA Corp. & Affiliates |
| RynnLAM | `rynnlam/` | Apache-2.0 | The RynnLAM Authors |
| └ DA3 / DINOv2 backbone | `rynnlam/rynnlam/modules/dinov2/` | Apache-2.0 | Meta Platforms; ByteDance |

## Where the texts live

- **cosmos-framework** — `third_party/cosmos-framework/LICENSE` (OpenMDW-1.1),
  `NOTICE` (HuggingFace Transformers / Qwen, Apache-2.0; TaylorSeer, GPL-3.0;
  and others), `ATTRIBUTIONS.md` (Python dependency notices).
- **cosmos_tokenizers** — `third_party/cosmos_tokenizers/edge/model_card/`
  (model card, including the BIAS / PRIVACY / SAFETY / EXPLAINABILITY documents,
  carried verbatim) and `LICENSE-OpenMDW-1.1.txt`. The bundled files are the
  **text tokenizer** (vocabulary) only.
- **RynnLAM** — `rynnlam/LICENSE` (Apache-2.0) and `rynnlam/NOTICE`
  (DA3/DINOv2: Meta Platforms and ByteDance, Apache-2.0; references to DINO,
  pytorch-image-models, CodeLlama, rope-vit retained there).

## Copyleft components

One bundled file is under a license other than Apache-2.0 or OpenMDW-1.1. It is
separable, is not used by RynnWorld-Latent's training or inference path, and
remains under its original license — nothing here relicenses it.

**GPL-3.0** — `third_party/cosmos-framework/cosmos_framework/model/generator/utils/taylorseer.py`,
inherited unchanged from NVIDIA's cosmos-framework and attributed in that
subtree's `NOTICE`. Upstream: <https://github.com/Shenyi-Z/TaylorSeer>.

## Composition-scan review

A third-party code-provenance / composition scan of this tree reported **88 files
with snippet similarity to upstream open-source projects, 0 vulnerabilities, and
0 embedded secrets/credentials**. Every flagged file lives either in the vendored
`third_party/cosmos-framework/` subtree or is a DINOv2 / RoPE / logger module now
bundled under `rynnlam/`.

The scan's "component" field names the *closest matching public repository*, which
is frequently a **downstream project that itself reused the same upstream code** —
not the actual licensor of the file. We verified the real license of each flagged
file from its in-file SPDX / copyright header. In every case the file is
**Apache-2.0 or NVIDIA OpenMDW-1.1**, both already covered by this notice set. No
AGPL, MPL, Commercial, or unlicensed code is actually present, so the non-Apache
matches below create **no additional copyleft obligation** on RynnWorld-Latent.

| Scan match (claimed license) | Flagged file(s) | **Actual license (verified header)** |
|---|---|---|
| TheAnimeScripter (AGPL-3.0) | `rynnlam/rynnlam/modules/dinov2/{vision_transformer,reference_view_selector}.py`, `dinov2/layers/` | Apache-2.0 — Meta Platforms / ByteDance (DINOv2) |
| SimpleTuner (AGPL-3.0) | `…/diffusion/samplers/fm_solvers_unipc.py` | OpenMDW-1.1 — NVIDIA (upstream diffusers, Apache-2.0) |
| libreyolo (MIT) | `rynnlam/rynnlam/modules/dinov2/dinov2.py` | Apache-2.0 — Meta Platforms (DINOv2) |
| open-oasis (MIT) | `…/modules/embeddings.py` (not present in current tree) | Apache-2.0 |
| FireRedTTS (MPL-2.0) | `…/tokenizers/audio/avae_utils/activations.py` | OpenMDW-1.1 — NVIDIA; file inline-attributes snake (MIT) |
| NVIDIA-NeMo/Speech (Commercial) | `…/easy_io/handlers/json_handler.py` | OpenMDW-1.1 — NVIDIA |
| Wan2GP (unknown) | `…/avae_utils/alias_free_torch/{act,filter,resample}.py`, `…/modules_encodec.py`, `flux_vae_8x8.py`; `rynnlam/.../rope.py`, `logger.py` | OpenMDW-1.1 — NVIDIA (alias-free-torch inline-attributed, Apache-2.0); rynnlam files Apache-2.0 |
| wow-world-model (no license) | `…/utils/{profiling,misc,config}.py`, `callbacks/{manual_gc,device_monitor}.py`, `trainer/__init__.py`, `…/imageio_video_handler.py` | OpenMDW-1.1 — NVIDIA |

Two of the flagged cosmos-framework files already carry their own upstream
attributions inline and are unchanged: `avae_utils/activations.py` (adapted from
`EdwardDixon/snake`, MIT) and `avae_utils/alias_free_torch/act.py` (adapted from
`junjun3518/alias-free-torch`, Apache-2.0). The GPL-3.0 `taylorseer.py` remains
the only genuine copyleft file and is separable and unused by the training /
inference path (see above).

**Per-file provenance headers.** Every file the scan matched at **>=80%**
similarity (57 files: 44 under `third_party/cosmos-framework/`, and 13 under
`rynnlam/` — the 12 DINOv2 files in `rynnlam/rynnlam/modules/dinov2/` plus
`rynnlam/rynnlam/logger.py`) now carries a
`# Provenance reference (composition scan ...)` comment block at the top. The
block records the scanner's nearest-repo match, states explicitly that this is
*not* a verified derivation, and gives the file's actual in-header license. It
prepends a comment only — no code is changed. These headers are one of a small
number of local modifications to the otherwise upstream-faithful vendored
subtree; the complete, authoritative list — including the one functional
(non-comment) edit, a `.removeprefix("net.")` fix in
`model/generator/utils/safetensors_loader.py` so the released safetensors weights
load — is in `NOTICE`, "Vendored subtree vs. upstream".

## Weights (downloaded, not bundled)

| Weight | Source | License |
|---|---|---|
| Cosmos3-Edge backbone | HF `nvidia/Cosmos3-Edge` (gated) | NVIDIA OpenMDW-1.1 |
| Wan2.2 video VAE | bundled in the Cosmos3 release | NVIDIA / Wan2.2 terms |

The OpenMDW-1.1 license text for the weights is the same
`third_party/cosmos-framework/LICENSE`. Redistribution of the weights is
governed by that license and the gated-access terms on the Hugging Face page;
this repository deliberately ships **code only**.

## Data

Training and evaluation data is not redistributed, with one exception: a small
set of in-house samples ships under `data/` so a fresh clone trains, post-trains
and evaluates out of the box — three `tianji_wuji_data` chunks under
`data/{videos,latents,manifest}` for the world-model quickstart, plus one real
record each for the two downstream post-train quickstarts (`data/astribot_s1/`
for Astribot-S1 and `data/marvin_wuji/` for Marvin-WUJI). Provenance and frame
windows are in the README. The
corpora this project was developed against (RynnVLA-Latent and its RynnLAM
latents, RoVid-X, EgoVerse, RoboMIND, AgiBot, DROID, EPIC-KITCHENS, EgoDex, and
the DreamDojo / GR00T-Teleop-GR1 evaluation sets) each carry their own access
terms and must be obtained from their publishers. See the README for what each
is used for.

## Relationship to RynnWorld-Latent

RynnWorld-Latent fine-tunes the NVIDIA Cosmos3-Edge backbone (via
cosmos-framework) and is conditioned on continuous latent actions produced by
RynnLAM. The project's own code under `rynnworld_latent/`, `scripts/` and
`configs/` is Apache-2.0.
