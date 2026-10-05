## Explainability

| Field | Response |
| :---- | :---- |
| Intended Application & Domain | World reasoning and generation for Physical AI. |
| Model Type | Mixture-of-Transformers architecture with two towers. One is an autoregressive model for Physical AI reasoning; the other is a diffusion model for Physical AI generation. |
| Intended Users | Physical AI developers, researchers, and practitioners building or evaluating autonomous vehicle, robotics, and world-generation workflows. |
| Output | Images, videos, audio, and action commands. |
| Tools used to evaluate datasets to identify synthetic data and ensure data authenticity. | Dataset provenance analysis, metadata validation, watermark and artifact detection, embedding-based clustering, heuristic quality checks, and model-assisted data validation pipelines are used to identify synthetic content patterns, assess dataset authenticity, and improve data quality during dataset curation. |
| Describe how the model works | Cosmos3 is an Omni world foundation model that generates texts, images, videos, audio, and action commands from combinations of text, images, videos, and action trajectory inputs. Input tokens from multiple modalities are packed into a shared sequence and processed by our mixture-of-transformer backbone with modality-specific output heads. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | None. |
| Technical Limitations | The model may not follow text, image, video, audio, or action trajectory inputs accurately in challenging cases, especially where the input contains complex scene composition, unusual camera motion, multiple interacting agents, low lighting, high motion blur, or fine-grained physical interactions. Generated outputs may contain temporal inconsistency, object morphing, inaccurate 3D structure, or implausible physical dynamics. Generated audio may not accurately render intelligible speech, or maintain strict temporal and semantic alignment with the visual context. |
| Verified to have met prescribed NVIDIA quality standards | Yes. |
| Performance Metrics | Video generation is measured using PAIBench-G, RBench, PhysicsIQ, and Artifical Analysis Image2Video benchmark. Image generation uses UniGenBench and Artifical Analysis Text2Image benchmark. For transfer evaluation, we use PAIBench-C and AVBench-C. Audio generation uses internal benchmarks. Action prediction uses metrics such as action MSE, Absolute Translation Error, Relative Translation Error, Relative Rotation Error, PSNR, and robotic task completion success rate. |
| Potential Known Risks | This model can generate synthetic media and may produce content that is offensive, unsafe, misleading, indecent, or unsuitable for a target deployment. Users should implement robust safety guardrails — including content filtering, abuse monitoring, and access controls — to reduce the risk of harmful outputs. Users are responsible for ensuring that their use of the model complies with all applicable laws and regulations, and for regularly reviewing and updating their guardrails as risks evolve. |
| Licensing | [OpenMDW1.1](https://openmdw.ai/)  |
