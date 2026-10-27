# Apple Silicon inference qualification

The inference APIs select CUDA, then MPS, then CPU, with explicit device
overrides. See [installation and usage](../INSTALL.md#apple-silicon-macos).
The global baseline is **PyTorch >= 2.14.1 and matching TorchVision >= 0.29.1**.
Existing PyTorch models and checkpoints retain their formats.
Validated source revision: `d976cc5ae620313b2d6577d1a9b4ea71f4dfe351`.
Contribution: [upstream PR #778](https://github.com/facebookresearch/sam2/pull/778).

## Native Hiera and workload

The baseline supports native MPS bicubic positional-grid interpolation and the
strided MaxPool2d query views used by Hiera. The former positional-grid CPU
fallback and MPS pooling copy have been removed. A public backbone-forward test
compares CPU/MPS outputs. The [JSON evidence](validation/apple-silicon-pytorch-2.14.1.json)
records the tested Hiera source hash.

Measured on 27 October 2026 with an Apple M3 Pro, 36 GiB unified memory,
macOS 27.2, native arm64 Python 3.12.16 and six CPU threads. The official
SAM 2.1 tiny checkpoint has SHA-256
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.

The workload uses the bundled 1800x1200 truck image, prompt `(500, 375)`, a fixed
8x8 automatic-mask grid (batch size four), and bedroom frames `00000.jpg` through
`00002.jpg` at 960x540. Two video objects receive positive prompts `(200, 300)`
and `(400, 150)`. Model input resolution remains 1024. CPU and MPS use float32,
including stored MPS video memory. `PYTORCH_ENABLE_MPS_FALLBACK` was unset.

Prospective gates remain image IoU >= 0.99, at least 90% of CPU automatic masks
matched one-to-one at IoU >= 0.95, every video object/frame/session IoU >= 0.98,
and live GPU allocation growth <= 16 MiB over three cleaned sessions. Each
backend exercises reset/reuse, default frame storage, explicit device frame
storage with CPU tracking-state offload, and explicit CPU frame storage. The
harness checks finite outputs, nonempty masks, object IDs and shapes.

| Check | Native Hiera result, PyTorch 2.14.1 |
| --- | --- |
| Image mask IoU | 1.0 |
| Automatic mask matching | 12/12 masks at IoU >= 0.95 |
| Minimum video IoU, all sessions | 0.999768 |
| Reset/reuse and all storage variants | Passed |
| Three-session live allocation growth | 0 bytes |
| Live allocation after each cleaned session | 279,851,264 bytes (model remains loaded) |
| MPS propagation throughput | 2.85–2.94 frames/s for this workload |
| Cached image prompt latency | 12–17 ms, excluding image embedding |
| Automatic mask generation | 2.45 s, one fixed-grid sample |

These are local regression measurements. They establish neither a performance
ceiling nor a controlled comparison with older implementations or releases.
Image embedding and warmup are reported separately. Driver allocations include
reusable caches and are not peak live-tensor memory.

Connected components use SciPy on CPU when the CUDA extension is unavailable or
inapplicable, preserving 8-connectivity, area maps and the caller's device.
Async JPEG input is normalized to float32 before transfer. MPS video frames
default to CPU storage; explicit choices remain authoritative.

The shared MP4 path/bytes loader uses PyAV, which has native Apple Silicon
wheels. Notebook and demo extras both request it. Generated colored clips test
RGB/frame order, source dimensions, resizing, normalization and CPU/MPS storage.
A [fresh real-checkpoint MP4 run](validation/apple-silicon-mp4.json) with native
Hiera and PyAV 19.0.2 used three frames re-encoded from the bundled bedroom MP4.
Both objects, all frames and reset/reuse passed; minimum CPU/MPS IoU was 0.999770.

## Upgrade and loading validation

PyTorch changed `torch.load` to restricted weight loading by default starting
with 2.6. Model and Hiera weights explicitly retain `weights_only=True`.
Generic checkpoint helpers default to that policy and expose `weights_only=False`
only as an explicit opt-in for trusted Python state. Trainer resume and internal
worker-object transport explicitly deserialize their full state; their checkpoint
sources, workers and shared directories must be trusted. Unrestricted pickle
loading can execute code. No global safety override is used. See
[PyTorch's serialization guidance](https://docs.pytorch.org/docs/2.14/notes/serialization.html#torch-load-with-weights-only-true).

The automated suite passed **91 tests**, with **six CUDA hardware skips** on
PyTorch 2.14.1/TorchVision 0.29.1/PyAV 19.0.2. It includes safe-default rejection,
trusted NumPy checkpoint loading, a real CPU Adam/GradScaler save/resume followed
by an equivalent optimizer update, and real two-rank CPU Gloo object exchange
through memory/filesystem gather and disk/memory broadcast. The installed wheel
also passed decoder/backbone regressions with the minimum PyAV 13.0.0.

A separate [full CPU training smoke](validation/pytorch-2.14.1-cpu-training.json)
loaded the tiny checkpoint into `SAM2Train`, with all 38,962,498 parameters
trainable, two synthetic 1024x1024 frames and one moving object. Canonical
focal/Dice/IoU/object-score losses produced finite core loss 0.0604674. Backward
produced finite gradients, including all 162 image-encoder gradient tensors;
one AdamW step changed parameters in the image encoder, temporal memory and
prompt/mask decoders. Peak process RSS was 6.01 GB. Canonical MOSE augmentations
and collation also passed for three seeds with finite normalized images,
boolean masks and preserved identities. This is CPU smoke evidence; it does not
qualify MPS training, CUDA training, a full training job or convergence.

Strict Mac editable installation, source-archive installation and wheel
installation succeeded in a fresh virtual environment. All 47 packaged Python
sources matched the checkout. The wheel loaded its packaged tiny configuration,
official checkpoint, CPU image prediction, native TorchVision NMS and connected
components from outside the repository. The source archive retains CUDA source
even when built without CUDA tooling. Python 3.10 dependency resolution passed
for macOS 14 arm64 and the Linux CPU PyTorch/TorchVision pair; runtime tests used
Python 3.12.16.

Notebook, demo and development extras installed together. The local demo loaded
the tiny model on MPS and passed health and GraphQL requests. Complete notebook
sessions and browser-driven demo interactions were not exercised. Formatting,
documentation links and Docker Compose configuration passed. Hadolint passed
at its error threshold, retaining warnings for unpinned apt/pip packages and
shell-form CMD.
The existing TorchScript transform emits a deprecation warning on PyTorch 2.14.1.

The demo image is the published `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime`.
It installs the complete package before copying the server, uses one installed
package location, and has a `dependencies` build target without checkpoint
fetches. A container virtual environment with system-site-package access reuses
the image's PyTorch and permits application installation under PEP 668. The
obsolete Conda ffmpeg path was removed. An actual Docker build/run
was unavailable: the local daemon was stopped and host free space was 11 GiB.

## Reproduction and limits

Run [the qualification command](../INSTALL.md#apple-silicon-macos) after installing
the tiny checkpoint. Tests run with `python -m pytest -q tests`; install `.[dev]`
and PyAV to include training-state and decoder contracts. Accelerator tests skip
when hardware is unavailable. Hosted Linux/macOS CI covers minimum/current
PyTorch and a separate CPU checkpoint workload. Hardware skips do not qualify an
accelerator.

CUDA extension compilation/execution, CUDA training, compiled VOS, larger
checkpoints, long clips and complete notebook sessions remain unqualified on
this Mac. Rebuild CUDA extensions against the upgraded PyTorch and matching
CUDA toolkit. No custom Metal kernel, MLX port, package release or upstream merge
is included.
