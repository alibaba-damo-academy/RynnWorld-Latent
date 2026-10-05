# Vendored Cosmos3 text tokenizer — origin and license notices

Contents : text_tokenizer/ of nvidia/Cosmos3-Edge (under edge/), plus that model's
           card documents under edge/model_card/.
           NO model weights are included (tokenizer files only, ~17 MB).
Origin   : the Hugging Face snapshot of nvidia/Cosmos3-Edge, the repository that
           cosmos-framework's own EDGE_MODEL_CONFIG names as the tokenizer source.
License  : OpenMDW License Agreement, version 1.1 — full text in LICENSE-OpenMDW-1.1.txt
           (identical to third_party/cosmos-framework/LICENSE). The model card states
           license_name: openmdw1.1-license and "ready for commercial and non-commercial use".
Conditions retained per OpenMDW-1.1: (1) a copy of the agreement travels with this
           distribution (LICENSE-OpenMDW-1.1.txt); (2) copyright and origin notices are
           retained (this file + edge/model_card/README.md).
Further notices of origin (BIAS / PRIVACY / SAFETY / EXPLAINABILITY statements) are
           carried verbatim under edge/model_card/.
Why here : cosmos-framework resolves the tokenizer through
           CheckpointDirHf(repository="nvidia/Cosmos3-Edge"), which shells out to a
           Hugging Face Hub download. Bundling the files lets training and inference
           run with no Hub access; rynnworld_latent/inference.py and scripts/train.py
           patch the processor to read this directory instead.
Override : set RYNNWORLD_EDGE_HF_DIR to point at another Cosmos3-Edge snapshot;
           the code prefers the env value over this bundled copy.
