"""RynnWorld-Latent: a video world model conditioned on RynnLAM latent actions.

Submodules are imported explicitly (``from rynnworld_latent.manifest_dataset import
RynnWorldManifestDataset``); this package deliberately has no eager imports, so
importing it costs nothing and pulls in neither torch nor cosmos_framework.
"""

__version__ = "0.1.0"
