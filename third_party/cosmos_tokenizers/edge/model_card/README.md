---
license: other
license_name: openmdw1.1-license
license_link: >-
  https://openmdw.ai/license/1-1/
library_name: cosmos
tags:
  - nvidia
  - cosmos
  - cosmos3
---

# **Cosmos 3: Omnimodal World Models for Physical AI**
**[Model Collection](https://huggingface.co/collections/nvidia/cosmos3)** | **[Code](https://github.com/nvidia/cosmos)** | **[White Paper](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)** | **[Website](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)**

[NVIDIA Cosmos™](https://github.com/nvidia/cosmos) is a world foundation model platform designed to accelerate the development of Physical AI by enabling machines to understand, simulate, and interact with the physical world across robotics, autonomous driving, and smart space environments, including industrial and factory-scale applications.

# Model Overview: Cosmos3-Edge

## Description

Cosmos3 is a collection of Omnimodal world models capable of generating dynamic, high-quality video, image, audio, and action commands from combinations of text, image, video, and action trajectory inputs. It serves as a foundational building block for a broad range of Physical AI applications and research spanning world understanding, world generation, simulation, and embodied policy learning.

This model is ready for commercial and non-commercial use.

**Model Developer:** NVIDIA

### Model Versions

**Released on: 07/20/2026**
- Cosmos3-Edge:
  - Given multimodal inputs including text, images, video, and action trajectories, generate coherent text, images, video, and action outputs for multimodal understanding, world simulation, future prediction, action reasoning, and Physical AI applications.

- Cosmos3-Edge-Policy-DROID:
  - Given language instructions and visual observations from the DROID robot platform, generate robot action trajectories for manipulation and control tasks.

- Cosmos3-Super-Image2Video-4Step:
  - Given one or more input images and optional text instructions, generate temporally coherent video sequences that are consistent with the provided visual content. Distilled from Cosmos3-Super-Image2Video using [Improved Distribution Matching Distillation (DMD2)](https://arxiv.org/abs/2405.14867), enabling high-quality generation in 4 steps.

- Cosmos3-Super-Text2Image-4Step:
  - Given text input, generate high-fidelity images that are consistent with the provided description. Distilled from Cosmos3-Super-Text2Image using [Improved Distribution Matching Distillation (DMD2)](https://arxiv.org/abs/2405.14867), enabling high-quality generation in 4 steps.

**Released on: 05/31/2026**
- Cosmos3-Nano:
  - Given multimodal inputs including text, images, video, audio, and action trajectories, generate coherent text, images, video, audio, and action outputs for multimodal understanding, world simulation, future prediction, action reasoning, and Physical AI applications.

- Cosmos3-Super:
  - Given multimodal inputs including text, images, video, audio, and action trajectories, generate coherent text, images, video, audio, and action outputs for multimodal understanding, world simulation, future prediction, action reasoning, and Physical AI applications.

- Cosmos3-Nano-Policy-DROID:
  - Given language instructions and visual observations from the DROID robot platform, generate robot action trajectories for manipulation and control tasks.

- Cosmos3-Super-Image2Video:
  - Given one or more input images and optional text instructions, generate temporally coherent video sequences that are consistent with the provided visual content.

- Cosmos3-Super-Text2Image:
  - Given text input, generate high-fidelity images that are consistent with the provided description.

### License

This model is released under the [OpenMDW1.1](https://openmdw.ai/license/1-1/)

### Deployment Geography

Global

### Use Case

Physical AI: Encompassing robotics, autonomous vehicles (AV), and smart space environments, including industrial and factory-scale applications.

### Release Date

Hugging Face 07/20/2026 via [https://huggingface.co/collections/nvidia/cosmos3](https://huggingface.co/collections/nvidia/cosmos3)
GitHub 07/20/2026 via [https://github.com/nvidia/cosmos](https://github.com/nvidia/cosmos)

## Model Architecture

**Architecture Type:** Transformer

**Network Architecture:** Mixture-of-Transformers (MoT)

Cosmos3 is an Omni-modal foundation model built on a Mixture-of-Transformers (MoT) architecture consisting of two complementary transformer towers: an autoregressive transformer for discrete token generation and a diffusion transformer for continuous multimodal generation. During inference, text is generated through standard next-token autoregressive decoding, while non-text modalities, such as images, video, audio, and actions, are synthesized through iterative denoising. This unified architecture enables Cosmos3 to model heterogeneous modalities within a single framework while preserving generation mechanisms best suited to each modality.

**This model was developed based on:**  [Cosmos Framework](https://github.com/nvidia/cosmos-framework)

**Number of trainable model parameters:**

**Released on: 07/20/2026**
- Cosmos3-Edge: 4B
- Cosmos3-Edge-Policy-DROID: 4B
- Cosmos3-Super-Image2Video-4Step: 64B
- Cosmos3-Super-Text2Image-4Step: 64B

**Released on: 05/31/2026**
- Cosmos3-Nano: 16B
- Cosmos3-Super: 64B
- Cosmos3-Nano-Policy-DROID: 16B
- Cosmos3-Super-Image2Video: 64B
- Cosmos3-Super-Text2Image: 64B

## Input/Output Specifications

- **Generator Input**
  - **Input Type(s)**: Text, Image, Action Trajectory
  - **Input Format(s)**:
    - Text: String
    - Image: jpg, png, jpeg, webp
    - Action: JSON — 2D array shaped (T, D), where T is the number of frames and D is the embodiment-specific dimensionality
  - **Input Parameters**:
    - Text: One-dimensional (1D)
    - Image: Two-dimensional (2D)
    - Action trajectory: Two-dimensional (2D)
  - **Other Properties Related to Input**:
    - Image input is RGB color (8 bits per channel, sRGB color space); grayscale inputs are not supported.
    - Action input is a per-frame sequence of robot/agent state or control values (e.g., joint positions, gripper state, camera pose). The full input is a 2D array shaped (T, D), where T is the number of frames and D is the embodiment-specific dimensionality listed below.
    - Input action is only supported for compatible embodiments, including general camera motion (9D), autonomous vehicle (9D), egocentric motion (57D), single Franka Panda arm with RobotiQ gripper (10D), dual Franka Panda arm with RobotiQ gripper (20D), Agibot (29D), UR (10D), Google robot (10D), WidowX 250 (10D), UMI (10D).
  - **Input Size and Length limits:**
    - **Text:** 4096 tokens
    - **Image:** 256p and 480p resolution at one of these aspect ratios (16:9, 4:3, 1:1, 3:4, 9:16)
    - **Action:** 16 – 400 sequence length
- **Generator Output**
  - **Output Type(s)**: Image, video, action, text
  - **Output Format(s)**:
    - Image: JPG
    - Video: MP4
    - Action: JSON
    - Text: string
  - **Output Parameters**:
    - Image: Two-dimensional (2D)
    - Video: Three-dimensional (3D)
    - Action: Two-dimensional (2D)
    - Text: One-dimensional (1D)
  - **Other Properties Related to Output**:
    - The generated video is an MP4 file, with the resolution, frame rate, and duration specified in the input.
    - Video generation supports 256p and 480p resolution, 12–30 fps, and 50–150 frames. These are set per request via the `size`, `fps`, and `num_frames` fields.
    - The generated action is only supported for compatible embodiments, including general camera motion (9D), autonomous vehicle (9D), egocentric motion (57D), single Franka Panda arm with RobotiQ gripper (10D), dual Franka Panda arm with RobotiQ gripper (20D), Agibot (29D), UR (10D), Google robot (10D), WidowX 250 (10D), UMI (10D).
    - Video: mp4 at the FPS specified in input
    - Image: JPEG
- **Reasoner Input**
  - **Input Type(s)**: Text, Text+Image, Text+Video
  - **Input Format(s)**:
    - Text: String
    - Image: jpg, png, jpeg, webp
    - Video: mp4
  - **Input Parameters**:
    - Text: One-dimensional (1D)
    - Image: Two-dimensional (2D)
    - Video: Three-dimensional (3D)
  - **Other Properties Related to Input**:
    - Video inputs are recommended at a frame rate of 4 fps.
    - Long-context inputs supported up to 256K tokens.
  - **Input Size and Length limits:**
    - **Text:** Up to 256K tokens (context window).
    - **Image:** Standard input image formats; passed as file or URL.
    - **Video:** mp4 at the recommended 4 fps.
- **Reasoner Output**
  - **Output Type(s)**: Text
  - **Output Format(s)**:
    - Text: string
  - **Output Parameters**:
    - Text: One-dimensional (1D)
  - **Other Properties Related to Output**:
    - Default `max_tokens=4096+` is recommended for reasoning outputs; longer outputs may be requested.
    - Reasoning outputs may include structured chain-of-thought, 2D/3D point localization, and bounding-box coordinates for vision-based tasks.

The video content visualizes the input text description as a short animated scene, capturing key elements within the specified time constraints.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g., GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

**Runtime Engine(s):**

- [PyTorch](https://github.com/nvidia/cosmos3)

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA Ampere
- NVIDIA Blackwell
- NVIDIA Hopper

**Operating System(s):**

- Linux (We have not tested on other operating systems.)

**Note:** Only BF16 precision is tested. Other precisions like FP4, FP8, and FP16 are not officially supported.

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Training, Testing, and Evaluation Datasets

### Dataset Overview

- **Total Size:** 1.3B data points
- **Total Number of Datasets:** 393 dataset entries
- **Dataset partition:** Training [100%], Testing [N/A — evaluation benchmarks used separately], Validation [N/A — evaluation benchmarks used separately]
- **Time period for training data collection:** 2024–2026
- **Time period for testing data collection:** N/A (standard public benchmarks)
- **Time period for validation data collection:** N/A (standard public benchmarks)

Raw data from internal and external sources is transformed into training-ready data through multiple stages of curation, filtering, and quality review. Data acquisition spans diverse multimodal sources — robotics, autonomous driving, industrial environments, indoor and outdoor scenes, varied lighting and weather conditions, camera viewpoints, object categories, and human activities — to broaden coverage across Physical AI operating environments. Automated filtering pipelines remove corrupted, duplicate, low-quality, and restricted content. Metadata analysis, heuristic rules, and model-assisted classifiers are applied during preprocessing to flag anomalous distributions and low-diversity subsets. Human review supplements automated filtering for selected datasets, benchmark construction, and targeted quality analysis. Datasets are balanced across modalities and task categories — visual reasoning, text-to-image, text-to-video, image-to-video, video transfer, action-conditioned generation, and action command generation — to reduce overrepresentation of narrow domains. Synthetic and simulation-based augmentation supplements coverage of rare physical interactions and edge-case scenarios. Deduplication and provenance tracking are applied across the corpus. The resulting processed data is converted into model-ready tokenized or encoded representations through modality-specific preprocessors before training begins.

Training datasets passed through multiple layers of automated and manual safeguards designed to reduce the presence of harmful or policy-violating content across categories including weapons and weapons-related instructional content, criminal planning, child sexual abuse material (CSAM), non-consensual intimate imagery (NCII), sexual content involving minors, harassment, hate speech, profanity, threats and incitement to violence, self-harm or suicide-related content, and graphic violence. Data sources are reviewed for licensing compatibility, provenance, and alignment with internal data governance and safety policies before admission into training corpora. Automated filtering pipelines combine multiple detection strategies: hash-matching against known CSAM and NCII reference databases; classifier-based moderation models trained for explicit sexual content, hate speech, violence, weapons imagery, and other restricted categories; keyword and regex-based screening for criminal-planning, threats, and self-harm phrases in text data; metadata and provenance heuristics for source-level risk signals; and embedding-based anomaly detection to surface samples that fall outside expected distributions. Human review and targeted audits supplement automated filtering for selected datasets, benchmark construction, and safety-sensitive evaluation. For multimodal Physical AI data (robotics, autonomous driving, industrial scenes), additional filtering targets invalid action trajectories, physically implausible interactions, and unsafe control sequences. Synthetic and simulation-generated data are evaluated through internal validation before inclusion. Benchmark evaluations and red-team testing are applied post-training to surface remaining safety gaps across world generation, reasoning, and action tasks. No large-scale data-filtering process can guarantee complete removal of all harmful content; residual risks may remain, particularly in rare edge cases or open-world deployment settings. Ongoing monitoring and dataset review continue post-release.

**Data Modality and Training Data Size**

| Modality | Reasoning Data Sample Count | Generation Data Sample Count |
| -------- | ------------------- | -------------------- |
| Text     | 22M                 | Not Applicable       |
| Image    | 19M                 | 767M                 |
| Video    | 1M                  | 348M                 |
| Action   | Not Applicable      | 7M                   |

**Data Collection Method by dataset**

- Hybrid: Automatic/Sensors, Synthetic, Automated

**Labeling Method by dataset**

- Hybrid: Human, Automated

**Properties:** The training, testing, and evaluation datasets consist of diverse multimodal video, image, action, synthetic, and sensor-conditioned data sourced from NVIDIA-owned data and publicly available, commercially permissive datasets. These datasets are curated to exclude known restricted content and to support building an Omni model that learns to generate and reason about dynamic physical environments across world reasoning and generation tasks.

### Public Datasets

| Dataset                   | Samples     |
|---------------------------|-------------|
| OpenImage                 | 1.2M        |
| Coyo700M                  | 100M        |
| YouTube Video             | 340M        |
| UMI                       | 4.5M        |

### Private Datasets

| Dataset                   | Samples     |
|---------------------------|-------------|
| Egocentric                | 7M          |
| Nexar                     | 0.6M        |
| AgiBot                    | 0.2M        |
| HOI                       | 0.3M        |

### Synthetic Datasets

| Dataset                                 | Samples     |
|-----------------------------------------|-------------|
| synthetic images generated using HiDream-I1       | 15M         |
| synthetic images generated using Qwen-Image-2512  | 14M         |
| synthetic captions generated using Qwen3-VL       | 1115M       |

## Evaluation Datasets

**Data Collection Method by dataset**

- Hybrid: Automatic/Sensors, Synthetic, Automated

**Labeling Method by dataset**

- Hybrid: Human, Automated

**Properties:** The training, testing, and evaluation datasets consist of diverse multimodal video, image, action, synthetic, and sensor-conditioned data sourced from NVIDIA-owned data and publicly available, commercially permissive datasets. These datasets are curated to exclude known restricted content and to support building an Omni model that learns to generate and reason about dynamic physical environments across world reasoning and generation tasks.

## Benchmarks

For detailed evaluations of the base model, see our [technical paper](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf).

### Overall

The table below summarizes Cosmos3-Edge across reasoning and generation. Each reasoning column (General, Robotics, Smart Infrastructure, Driving) reports the average score over that capability's benchmarks. For generation, **Image2Video** is the PAIBench overall score and **Policy: Robot** is the RoboLab success rate. In each column, the best result is in **bold** and the second-best is <u>underlined</u>.
\* denotes post-trained Cosmos3 variants: [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) and [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID#robolab).

![Overall benchmark results](assets/benchmark-overall.png)

### Reasoning Benchmarks

![Reasoning benchmarks](images/benchmark-reasoning.png)

### Generation Benchmarks

#### Image-to-Video Generation

All models are evaluated on image-to-video generation at **480p, 24 fps**. Throughput is the number of frames generated per second, measured in eager mode on a single **NVIDIA H100** GPU. Cosmos3-Edge delivers the highest generation throughput while achieving competitive quality across PAIBench, RBench, and PhysicsIQ.

![Image-to-Video benchmark results](assets/benchmark-image2video.png)


#### Action

The Edge model is a strong initialization for downstream action tasks. For example, post-training it on the DROID dataset produces a policy whose RoboLab success rate is reported in the [Cosmos3-Edge-Policy-DROID model card](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID#robolab).

### PBR (Performance Benchmark Reporting)

The following tables report single-GPU or single-platform inference performance for the Cosmos3-Edge **Generator** and **Reasoner** towers.

Generator results are measured using end-to-end or generation latency in seconds; lower is better. Reasoner results include serving and token-generation metrics, such as time to first token, request latency, and throughput.

All results were measured using a single GPU and a batch size of 1.

#### Generator

Unless otherwise noted, visual-generation benchmarks use **480p resolution**. Image-to-video benchmarks generate **189 frames**.

##### vLLM-Omni

| GPU or Platform | Image-to-Video | Forward Dynamics | Inverse Dynamics | Policy DROID |
| --- | ---: | ---: | ---: | ---: |
| B200 SXM 192 GB | — | 2.44 s | 3.98 s | 0.99 s |
| H100 SXM 80 GB | 27.64 s | 3.91 s | 5.60 s | 1.41 s |
| H100 NVL 96 GB | 35.60 s | 4.73 s | 6.39 s | 1.37 s |
| H20 SXM 96 GB | 108.16 s | 12.77 s | 15.49 s | 3.41 s |
| RTX PRO 6000 Blackwell Server Edition | 36.29 s | 5.65 s | 7.46 s | 1.87 s |
| DGX Station | 12.17 s | 4.33 s | 6.34 s | 8.11 s |
| DGX Spark | 165.96 s | 26.43 s | 30.86 s | 7.66 s |
| Jetson AGX Thor T5000, 128 GB, MAXN | 137.50 s | 6.05 s | 7.19 s | 6.32 s |
| Jetson T3000, 32 GB, 1100 MHz | 194.76 s | 8.67 s | 10.25 s | 8.63 s |
| Jetson T2000, 16 GB, 702 MHz, THOR_NANO | 101.20 s | — | — | — |

##### PyTorch

| GPU or Platform | Image-to-Video | Forward Dynamics | Inverse Dynamics | Policy DROID |
| --- | ---: | ---: | ---: | ---: |
| H100 SXM 80 GB | 23.92 s | 3.69 s | 3.56 s | 1.25 s |
| H100 NVL 96 GB | 32.24 s | 4.64 s | 4.52 s | 1.28 s |
| H20 SXM 96 GB | 97.51 s | 12.78 s | 12.64 s | 2.92 s |
| RTX PRO 6000 Blackwell Server Edition | 38.98 s | 5.26 s | 5.66 s | 1.32 s |
| DGX Station | 10.57 s | 2.16 s | 2.26 s | 1.30 s |
| DGX Spark | 179.80 s | 24.59 s | 26.76 s | 5.44 s |
| Jetson AGX Thor T5000, 128 GB, MAXN | 153.00 s | — | — | — |
| Jetson T3000, 32 GB, 1100 MHz | 227.80 s | — | — | — |

#### Reasoner

The following tables report Cosmos3-Edge Reasoner performance. Reasoner workloads produce autoregressively generated text and therefore use different metrics from the Generator workloads:

- **Time To First Token (TTFT):** Time from request submission until the first output token is produced. Lower is better.
- **Request Latency:** End-to-end latency for the complete request. Lower is better.
- **Request Throughput:** Completed requests per second. Higher is better.
- **Output Token Throughput:** Generated output tokens per second. Higher is better.

##### vLLM serving benchmarks

These measurements use the `nvidia/Cosmos3-Edge` checkpoint with one GPU. Metrics were collected at client-side concurrency levels of 1, 64, 128, and 256.

The workload notation is **input sequence length / output sequence length / video FPS**.

###### RTX PRO 4500 Blackwell Server Edition

| Input / Output / Video FPS | Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
| --- | --- | ---: | ---: | ---: | ---: |
| 50 / 1 / 1 | Time To First Token (ms) ↓ | 165.79 | 8817.33 | 14702.20 | 29482.39 |
|  | Request Latency (ms) ↓ | 165.79 | 8817.33 | 14702.20 | 29482.39 |
|  | Request Count (requests) | 50 | 320 | 256 | 512 |
|  | Request Throughput (req/s) ↑ | 6.00 | 6.55 | 6.55 | 6.52 |
|  | Output Token Throughput (tok/s) ↑ | 6.00 | 6.55 | 6.55 | 6.52 |
| 50 / 1 / 2 | Time To First Token (ms) ↓ | 371.67 | 20375.98 | 33812.45 | 68201.55 |
|  | Request Latency (ms) ↓ | 371.67 | 20375.98 | 33812.45 | 68201.55 |
|  | Request Count (requests) | 50 | 313 | 249 | 492 |
|  | Request Throughput (req/s) ↑ | 2.68 | 2.77 | 2.76 | 2.71 |
|  | Output Token Throughput (tok/s) ↑ | 2.68 | 2.77 | 2.76 | 2.71 |
| 50 / 100 / 1 | Time To First Token (ms) ↓ | 166.86 | 6900.90 | 19625.83 | 45729.55 |
|  | Request Latency (ms) ↓ | 764.15 | 16667.01 | 29196.84 | 55749.62 |
|  | Request Count (requests) | 50 | 320 | 256 | 512 |
|  | Request Throughput (req/s) ↑ | 1.31 | 3.73 | 3.74 | 3.70 |
|  | Output Token Throughput (tok/s) ↑ | 130.63 | 372.40 | 373.98 | 369.87 |
| 50 / 100 / 2 | Time To First Token (ms) ↓ | 374.93 | 23526.65 | 47550.99 | 101553.31 |
|  | Request Latency (ms) ↓ | 1041.29 | 33712.54 | 57641.53 | 111895.20 |
|  | Request Count (requests) | 50 | 320 | 256 | 512 |
|  | Request Throughput (req/s) ↑ | 0.96 | 1.79 | 1.79 | 1.78 |
|  | Output Token Throughput (tok/s) ↑ | 95.74 | 178.73 | 178.89 | 178.15 |

###### RTX PRO 6000 Blackwell Server Edition

| Input / Output / Video FPS | Metric | Concurrency 1 | Concurrency 64 | Concurrency 128 | Concurrency 256 |
| --- | --- | ---: | ---: | ---: | ---: |
| 50 / 1 / 1 | Time To First Token (ms) ↓ | 141.99 | 3213.91 | 5384.51 | 10792.72 |
|  | Request Latency (ms) ↓ | 141.99 | 3213.91 | 5384.51 | 10792.72 |
|  | Request Count (requests) | 50 | 320 | 254 | 512 |
|  | Request Throughput (req/s) ↑ | 6.96 | 18.00 | 17.95 | 17.89 |
|  | Output Token Throughput (tok/s) ↑ | 6.96 | 18.00 | 17.95 | 17.89 |
| 50 / 1 / 2 | Time To First Token (ms) ↓ | 239.86 | 7483.22 | 12552.69 | 25259.11 |
|  | Request Latency (ms) ↓ | 239.86 | 7483.22 | 12552.69 | 25259.11 |
|  | Request Count (requests) | 49 | 303 | 249 | 491 |
|  | Request Throughput (req/s) ↑ | 4.06 | 7.28 | 7.49 | 7.34 |
|  | Output Token Throughput (tok/s) ↑ | 4.06 | 7.28 | 7.49 | 7.34 |
| 50 / 100 / 1 | Time To First Token (ms) ↓ | 138.74 | 943.46 | 2680.17 | 11599.63 |
|  | Request Latency (ms) ↓ | 503.44 | 6188.90 | 13022.07 | 26388.89 |
|  | Request Count (requests) | 50 | 320 | 256 | 512 |
|  | Request Throughput (req/s) ↑ | 1.98 | 10.27 | 9.57 | 8.95 |
|  | Output Token Throughput (tok/s) ↑ | 197.75 | 1026.14 | 956.47 | 893.91 |
| 50 / 100 / 2 | Time To First Token (ms) ↓ | 239.24 | 1798.96 | 11644.84 | 33293.32 |
|  | Request Latency (ms) ↓ | 638.71 | 13599.89 | 26299.90 | 49165.91 |
|  | Request Count (requests) | 50 | 320 | 256 | 512 |
|  | Request Throughput (req/s) ↑ | 1.56 | 4.66 | 4.50 | 4.45 |
|  | Output Token Throughput (tok/s) ↑ | 155.93 | 465.28 | 449.57 | 444.17 |

##### Embedded-platform eager Transformers benchmarks

These preliminary measurements use raw Hugging Face Transformers in eager mode rather than vLLM. They are presented separately because their runtime, workload, and metric definitions differ from the vLLM serving benchmarks above.

| Board | Specification | Input | Prompt Tokens | Prefill Throughput | Prefill Latency | Decode Throughput | E2E Latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Jetson AGX Thor T5000 | 128 GB / MAXN | Text | 1705 | 8717 tok/s | 0.20 s | 37.3 tok/s | 3.60 s |
| Jetson AGX Thor T5000 | 128 GB / MAXN | Image | 911 | 4845 tok/s | 0.19 s | 42.6 tok/s | 3.17 s |
| Jetson AGX Thor T5000 | 128 GB / MAXN | Video | 1263 | 6032 tok/s | 0.21 s | 41.8 tok/s | 3.25 s |
| Jetson AGX Thor T4000 | 64 GB / MAXN, 1530 MHz | Text | 1705 | 6519 tok/s | 0.26 s | 34.1 tok/s | 3.99 s |
| Jetson AGX Thor T4000 | 64 GB / MAXN, 1530 MHz | Image | 911 | 3471 tok/s | 0.26 s | 40.3 tok/s | 3.41 s |
| Jetson AGX Thor T4000 | 64 GB / MAXN, 1530 MHz | Video | 1263 | 4164 tok/s | 0.30 s | 38.1 tok/s | 3.64 s |
| Jetson Thor T3000 | 32 GB / 1100 MHz | Text | 1705 | 5230 tok/s | 0.33 s | 29.7 tok/s | 4.61 s |
| Jetson Thor T3000 | 32 GB / 1100 MHz | Image | 911 | 2710 tok/s | 0.34 s | 36.3 tok/s | 3.83 s |
| Jetson Thor T3000 | 32 GB / 1100 MHz | Video | 1263 | 3388 tok/s | 0.37 s | 33.7 tok/s | 4.14 s |
| Jetson Thor T2000 | 16 GB / 702 MHz, THOR_NANO | Text | 1705 | 2355 tok/s | 0.72 s | 15.7 tok/s | 8.80 s |
| Jetson Thor T2000 | 16 GB / 702 MHz, THOR_NANO | Image | 911 | 1233 tok/s | 0.74 s | 19.6 tok/s | 7.21 s |
| Jetson Thor T2000 | 16 GB / 702 MHz, THOR_NANO | Video | 1263 | 1543 tok/s | 0.82 s | 18.0 tok/s | 7.87 s |
| Jetson AGX Orin | 64 GB | Text | 1705 | 3260 tok/s | 0.52 s | 12.3 tok/s | 10.83 s |
| Jetson AGX Orin | 64 GB | Image | 911 | 1840 tok/s | 0.50 s | 12.3 tok/s | 10.81 s |
| Jetson AGX Orin | 64 GB | Video | 1263 | 2103 tok/s | 0.60 s | 12.2 tok/s | 10.97 s |

**Benchmark notes**

- All Generator measurements use one GPU or one integrated computing platform.
- Generator values are average end-to-end or generation latency in seconds; lower is better.
- Unless otherwise specified, visual-generation measurements use **480p resolution**.
- Image-to-video measurements generate **189 output frames**.
- Jetson AGX Thor T5000 and Jetson T3000 visual-generation measurements use **832 × 480** resolution.
- Jetson T2000 visual-generation measurements use **448 × 256** resolution and therefore should not be compared directly with the 480p results. Its image-to-video values are warm-run measurements generating 189 frames.
- PyTorch Generator values report average generation latency rather than diffusion-only latency.
- Datacenter and enterprise forward- and inverse-dynamics results use the autonomous-driving (`AV`) configuration.
- Jetson AGX Thor T5000 and Jetson T3000 forward-dynamics, inverse-dynamics, and policy measurements use the DROID configuration with action chunk `[16, 8]`.

## Usage

- See [Cosmos](https://github.com/nvidia/cosmos) for details.

### Prompt upsampling

For optimal quality, prompts should be upsampled into a specific JSON structure. Description and code can be found [here](https://github.com/nvidia/cosmos-framework/blob/main/docs/prompt_upsampling.md).

For example, for image-to-video upsampling using Opus-4.6:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos-framework
pip install -e packages/cosmos-framework

export PROMPT_UPSAMPLER_ENDPOINT_URL="https://api.anthropic.com/v1/"
export PROMPT_UPSAMPLER_MODEL_NAME="claude-opus-4-6"
export PROMPT_UPSAMPLER_API_TOKEN="<your_token>"

python -m cosmos_framework.inference.prompt_upsampling \
    --input inputs/prompt_upsampler/prompts_i2v.txt \
    --image-list inputs/prompt_upsampler/images.txt \
    --output outputs/prompt_upsampler/upsampled_i2v_prompts_opus \
    --mode image2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 480 \
    --aspect-ratio "16,9" \
    --duration "5s" \
    --fps 24
```

For image-to-video, provide either one shared image via `--image-url` or one image per prompt via `--image-list` (the image-list file must have the same number of non-empty lines as the prompt file). Accepted image formats: local paths, HTTP(S) URLs, and `data:` URLs.

### Setup

Install cosmos-framework:

```shell
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget

git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework

# Pick the dependency group that matches your CUDA toolkit (see docs/setup.md)
uv sync --all-extras --group=cu130-train   # or --group=cu128-train for CUDA 12.8
source .venv/bin/activate && export LD_LIBRARY_PATH=
```

Cosmos3-Edge is a compact omni model that fits on a single GPU and is the recommended starting point for single-GPU inference. The examples below run offline batch inference via `cosmos_framework.scripts.inference`. Each example is self-contained and points at the same conditioning inputs used in this model card (hosted under the repo's `assets/`), so the shown outputs correspond to the shown inputs. Each sample's outputs are written under `-o <output_dir>/<sample_name>/` (e.g. `vision.mp4`, `sample_outputs.json`, `reasoner_text.txt`).

### Generation (image / video / action)

#### Image to video

```shell
BASE="https://huggingface.co/nvidia/Cosmos3-Edge/resolve/main/assets"
python - "$BASE" <<'PY'
import json, sys, urllib.request
base = sys.argv[1]
prompt = urllib.request.urlopen(f"{base}/example_i2v_prompt.json").read().decode()
json.dump(
    {"model_mode": "image2video", "prompt": prompt, "vision_path": f"{base}/example_i2v_input.jpg"},
    open("i2v_edge.json", "w"), indent=2,
)
PY

python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i i2v_edge.json \
    -o outputs/omni_edge \
    --checkpoint-path Cosmos3-Edge \
    --resolution 480 \
    --num-frames 121 \
    --fps 24 \
    --num-steps 50 \
    --guidance 5.0 \
    --shift 3.0 \
    --sampler unipc \
    --seed 0
```

The generated video is written to `outputs/omni_edge/<sample_name>/vision.mp4`.

Example output:

<video controls width="832" height="480" src="https://huggingface.co/nvidia/Cosmos3-Edge/resolve/main/assets/edge_i2v_output.mp4"></video>

#### Action — forward dynamics

Forward dynamics rolls out one action chunk at a time. This self-contained script fetches the UMI conditioning frame and action chunks, runs `cosmos_framework.scripts.inference` per chunk, feeds each chunk's final generated frame in as the next chunk's conditioning image, and stitches the 2-chunk rollout. Install `imageio[ffmpeg]` first.

```python
import json, subprocess, urllib.request
from pathlib import Path

import imageio.v3 as iio
import numpy as np

BASE = "https://huggingface.co/nvidia/Cosmos3-Edge/resolve/main/assets"
work = Path("umi_fd"); work.mkdir(exist_ok=True)

# 1. Fetch the UMI conditioning frame + action chunks from the model repo
first_frame = work / "first_frame.png"
urllib.request.urlretrieve(f"{BASE}/example_action_fd_umi_first_frame.png", first_frame)
spec = json.loads(urllib.request.urlopen(f"{BASE}/example_action_fd_umi_action_chunks.json").read())
chunks = spec["action_chunks"]            # shape (2, 16, 10)

current_frame, stitched = first_frame, []
for i, chunk in enumerate(chunks):
    # 2a. Write this chunk as a flat (16, 10) action_path file
    action_path = work / f"action_{i}.json"
    action_path.write_text(json.dumps(chunk))
    # 2b. Build the forward_dynamics input for this chunk
    #     (fields match inputs/omni/action_forward_dynamics_*.json)
    sample = {
        "name": f"umi_fd_{i}",
        "model_mode": "forward_dynamics",
        "domain_name": spec["domain_name"],              # "umi"
        "view_point": spec["view_point"],                # "ego_view"
        "fps": spec["fps"],                              # 20
        "image_size": spec["image_size"],                # 256
        "action_chunk_size": spec["action_chunk_size"],  # 16
        "prompt": spec["prompt"],                        # "mouse arrangement"
        "vision_path": str(current_frame),
        "action_path": str(action_path),
        "seed": i,
    }
    sample_path = work / f"sample_{i}.json"
    sample_path.write_text(json.dumps(sample))
    # 2c. Run cosmos-framework inference for this chunk
    subprocess.run([
        "python", "-m", "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        "-i", str(sample_path),
        "-o", str(work / "out"),
        "--checkpoint-path", "Cosmos3-Edge",
    ], check=True)
    # 2d. Read the rollout; drop the conditioning frame (index 0), keep the generated frames
    frames = iio.imread(work / "out" / f"umi_fd_{i}" / "vision.mp4")
    stitched.extend(frames[1:])
    # 2e. Autoregressive conditioning: last generated frame -> next chunk's input image
    next_frame = work / f"ar_frame_{i + 1}.png"
    iio.imwrite(next_frame, frames[-1])
    current_frame = next_frame

# 3. Stitch the generated-only rollout
out_mp4 = work / "umi_fd_2chunk.mp4"
iio.imwrite(out_mp4, np.asarray(stitched), fps=spec["fps"])
print("Saved stitched rollout:", out_mp4)
```

Example output:

<video width="512" controls src="https://huggingface.co/nvidia/Cosmos3-Edge/resolve/main/assets/edge_action_fd_umi_2chunk_output.mp4"></video>

#### Action — inverse dynamics

Inverse dynamics predicts the action sequence from an observation video (no action input). Repeat with `example_action_id_av_1_input.mp4` (and `name: av_inverse_1`) for the second example.

```shell
cat > av_inverse_0.json <<'JSON'
{
    "name": "av_inverse_0",
    "model_mode": "inverse_dynamics",
    "domain_name": "av",
    "view_point": "ego_view",
    "image_size": 480,
    "fps": 10,
    "action_chunk_size": 60,
    "prompt": "You are an autonomous vehicle planning system.",
    "vision_path": "https://huggingface.co/nvidia/Cosmos3-Edge/resolve/main/assets/example_action_id_av_0_input.mp4"
}
JSON

python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i av_inverse_0.json \
    -o outputs/omni_edge \
    --checkpoint-path Cosmos3-Edge \
    --seed 0
```

The predicted action is written to `outputs/omni_edge/<sample_name>/sample_outputs.json`.

Example outputs (predicted action trajectory, 60 steps × 9 raw action dims = translation + rot6d):

- [av_inverse_0 predicted action JSON](assets/edge_action_id_av_0_output.json)
- [av_inverse_1 predicted action JSON](assets/edge_action_id_av_1_output.json)

The predicted per-frame relative poses are integrated into an absolute camera trajectory (`pose_rel_to_abs`, `backward_framewise`) and visualized below — 3D path with camera frustums (left) and top-down bird's-eye (right), colored by frame index:

<img width="1280" src="assets/edge_action_id_av_0_output.png">

<img width="1280" src="assets/edge_action_id_av_1_output.png">

### Reasoning

The reasoner generates text from a prompt and an optional image or video (`vision_path`); output is written to `reasoner_text.txt`.

Image input:

<img src="assets/example_reasoning_input.png" alt="Reasoning example input showing a tabletop robot-manipulation scene" width="640">

User prompt:

```text
The task is to put flower into the red bottle. Generate a plan consisting of subtasks for accomplish the task.
```

```shell
cat > reasoner_edge.json <<'JSON'
{
    "model_mode": "reasoner",
    "prompt": "The task is to put flower into the red bottle. Generate a plan consisting of subtasks for accomplish the task.",
    "vision_path": "https://huggingface.co/nvidia/Cosmos3-Edge/resolve/main/assets/example_reasoning_input.png"
}
JSON

python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i reasoner_edge.json \
    -o outputs/reasoner_edge \
    --checkpoint-path Cosmos3-Edge \
    --seed 0
```

The generated text is written to `outputs/reasoner_edge/<sample_name>/reasoner_text.txt`.

Example output:

```text
<think>
Got it, let's try to figure out how to put the flower into the red bottle. First, I need to identify the objects: the flower is the green one with the red center, and the red bottle is the red container. So the steps would involve moving the flower to the red bottle.

First, the robot arm needs to locate the flower. Then, grasp the flower. After grasping, lift it, then move it towards the red bottle. Then, position the flower over the red bottle's opening, and finally release it into the bottle. Wait, but maybe I need to check if the red bottle has an opening that can accept the flower. Assuming the red bottle is a container with an opening, so the steps would be: move to flower, grasp, lift, move to red bottle, position over opening, release. Let me make sure each step is clear.

First subtask: Position the robot arm above the flower. Then, lower the gripper to grasp the flower. Then, lift the flower from the table. Then, move the robot arm with the flower towards the red bottle. Then, position the flower over the red bottle's opening. Then, release the flower into the red bottle. That should cover the task.</think>
To accomplish the task of putting the flower into the red bottle, the plan should follow these subtasks:

1. **Position the robot arm above the flower** – Move the robot arm to a position directly over the green flower with the red center.
2. **Lower the gripper to grasp the flower** – Lower the gripper to make contact with the flower and close it around the stem.
3. **Lift the flower from the table** – Once the flower is securely grasped, lift it upward to clear the table surface.
4. **Move the robot arm with the flower toward the red bottle** – Transport the flower (still held by the gripper) from its current position to the location of the red bottle.
5. **Position the flower over the red bottle’s opening** – Align the flower so its stem is directly above the opening of the red bottle.
6. **Release the flower into the red bottle** – Open the gripper to let the flower fall into the red bottle, completing the task.

Each step ensures the flower is picked up, transported, and placed into the red bottle in a controlled manner.
```

Thinking is enabled by default. Disabling it requires setting the reasoner chat template's `enable_thinking=False`, which the offline CLI does not currently expose — use the online serving path (`chat_template_kwargs={"enable_thinking": False}`) for a no-think response.

Guardrails are enabled by default (sourced from `nvidia/Cosmos-Guardrail1`); pass `--no-guardrails` to disable, or `--offload-guardrail-models` to keep them on CPU. For multi-GPU recipes, online Ray serving, and the full argument reference, see cosmos-framework `docs/inference.md`.

## Limitations

Cosmos3 may produce imperfect outputs in challenging scenarios. Generation artifacts include temporal inconsistency, unstable camera or object motion, imprecise physical interactions, and action-state drift — especially in long-horizon or high-resolution outputs. Reasoning may also be incorrect: object states, causal relationships, spatial geometry, temporal ordering, agent intent, and future outcomes can be misinferred, and complex or long-context inputs may yield hallucinated entities, inconsistent interpretations, or implausible predictions. Because the model lacks an explicit physics simulator, 3D geometry, 4D space-time evolution, object permanence, contact dynamics, and physical laws are only approximated — producing artifacts such as disappearing or morphing objects, unrealistic collisions, and physically implausible motions. Quality further degrades in out-of-distribution environments, safety-critical edge cases, and domains underrepresented in training.

Cosmos3 outputs should not be treated as physically accurate simulation, reliable ground-truth reasoning, or safety-certified decision making. Applications involving robotics control, autonomous systems, scientific simulation, or safety-critical planning require additional validation, external constraints, system-level safety analysis, and domain-specific guardrails before deployment.

## Inference

**Acceleration Engine:** [PyTorch](https://pytorch.org/)

**Test Hardware:** B200, H100, H20, RTX PRO 6000, DGX Station, DGX Spark, Jetson Thor, Jetson AGX Orin

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Users are responsible for model inputs and outputs. Users are responsible for ensuring safe integration of this model, including implementing guardrails as well as other safety mechanisms, prior to deployment.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Explainability](EXPLAINABILITY.md), [Bias](BIAS.md), [Safety & Security](SAFETY.md), and [Privacy](PRIVACY.md) subcards. Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
