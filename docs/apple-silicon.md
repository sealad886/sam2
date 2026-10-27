# Apple Silicon inference qualification

The inference APIs select CUDA, then MPS, then CPU, with explicit device
overrides. See [installation and usage](../INSTALL.md#apple-silicon-macos).
Existing PyTorch models and checkpoints are retained. Training is outside this
inference qualification.

## Workload and gates

Measured on 27 October 2026 with an Apple M3 Pro, 36 GiB unified memory,
macOS 27.2, native arm64 Python 3.12.16 and six CPU threads. The official
SAM 2.1 tiny checkpoint has SHA-256
`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.

The workload uses the bundled 1800x1200 truck image, prompt `(500, 375)`, a fixed
8x8 automatic-mask grid (batch size four), and bedroom frames `00000.jpg` through
`00002.jpg` at 960x540. Two video objects receive positive prompts `(200, 300)`
and `(400, 150)`. Model input resolution remains 1024. CPU float32 is the
reference; MPS uses float32, including stored video memory.

Before measurement, the delivery plan specified image mask IoU >= 0.99,
at least 90% of CPU automatic masks matched one-to-one to MPS masks at IoU >=
0.95, every video object/frame/session IoU >= 0.98, and live GPU allocation
growth <= 16 MiB over three cleaned sessions. Each backend also exercises
reset/reuse, default frame storage, explicit device frame storage with CPU
tracking-state offload, and explicit CPU frame storage. The harness checks
finite outputs, nonempty masks, object IDs and shapes. Global
`PYTORCH_ENABLE_MPS_FALLBACK` was unset for qualification.

## PyTorch 2.5.1 results

[Full machine-readable evidence](validation/apple-silicon-pytorch-2.5.1.json)
includes environment, settings, synchronized timings, per-session comparisons
and memory readings.

| Check | Observed result |
| --- | --- |
| Image mask IoU | 1.0 |
| Automatic mask matching | 12/12 masks at IoU >= 0.95 |
| Minimum video IoU, all sessions | 0.996519 |
| Reset/reuse and all storage variants | Passed |
| Three-session live allocation growth | 0 bytes |
| Live allocation after each cleaned session | 460,949,760 bytes (model remains loaded) |
| MPS propagation throughput | 2.63–2.86 frames/s for this two-object, three-frame workload |
| Cached image prompt latency | 22–34 ms, excluding image embedding |
| Automatic mask generation | 3.55 s, one fixed-grid sample |

These are local regression measurements, not a performance ceiling or a dataset
accuracy score. Image embedding and warmup are reported separately in JSON.
Driver allocations include reusable caches and are not peak live-tensor memory.
Other work on the machine can affect timings.

## PyTorch 2.14.1 results

The current release was independently exercised using the same workload,
checkpoint and gates. [Full evidence](validation/apple-silicon-pytorch-2.14.1.json)
records every measured storage variant.

| Check | Observed result |
| --- | --- |
| Image mask IoU | 1.0 |
| Automatic mask matching | 12/12 masks at IoU >= 0.95 |
| Minimum video IoU, all sessions | 0.999768 |
| Reset/reuse and all storage variants | Passed |
| Three-session live allocation growth | 0 bytes |
| Live allocation after each cleaned session | 280,506,624 bytes (model remains loaded) |
| MPS propagation throughput | 1.68–1.71 frames/s for this workload |
| Cached image prompt latency | 17–21 ms, excluding image embedding |
| Automatic mask generation | 5.73 s, one fixed-grid sample |

Timing samples came from separate local runs; they do not establish a controlled
performance comparison between PyTorch releases. PyTorch 2.14.1 also emits a
deprecation warning for the existing `torch.jit.script` transform.

## Confirmed backend corrections

Two failures were reproduced with the minimum supported PyTorch version:

- MPS MaxPool2d misread the noncontiguous query view created by QKV unbinding
  during Hiera stage transitions. Making only MPS pooling inputs contiguous
  reduced maximum error in a small public backbone-forward regression from
  about 0.186 to below 1e-6. CUDA/CPU pooling paths remain unchanged.
- MPS lacked bicubic interpolation. The backbone catches that specific
  unsupported operation and interpolates the small positional embedding grid
  on CPU, then returns it to MPS. MPS builds with native support keep that path.

Connected components use SciPy on CPU when the CUDA extension is unavailable or
inapplicable, preserving 8-connectivity, area maps and the caller's device.
Async JPEG input is normalized to float32 before transfer. MPS video frames
default to CPU storage; explicit choices remain authoritative.

The former optional `eva-decord` package had no Apple Silicon wheel. The shared
MP4 path/bytes loader now uses PyAV, already a demo dependency, and both notebook
and demo extras request it. Generated colored MP4 clips test RGB channel order,
frame order, source dimensions, resizing, float32 normalization and CPU/MPS frame
storage for paths and bytes. Extras dependency resolution was checked on arm64.

A [separate real-checkpoint MP4 run](validation/apple-silicon-mp4.json) on
PyTorch 2.14.1/PyAV 19.0.2 used three frames re-encoded from the bundled bedroom
MP4. Both objects, all frames and reset/reuse passed, with minimum CPU/MPS mask
IoU 0.999550. This is separate from the JPEG workload above.

The final automated suite passed **82 tests**, with **six CUDA-only hardware
skips**, on both PyTorch 2.5.1/PyAV 13.0.0 and PyTorch 2.14.1/PyAV 19.0.2.
Formatting and local documentation links were checked. The running web demo
and complete notebook sessions were not exercised.

A strict Mac editable installation and a CUDA-free wheel build succeeded. Every
Python source in the wheel matched the checkout. Importing that wheel from
outside the repository loaded its packaged tiny configuration, official
checkpoint and CPU connected-component cleanup successfully.

## Reproduction and limits

Run [the qualification command](../INSTALL.md#apple-silicon-macos) on the target
Mac after installing the tiny checkpoint. Unit tests run with
`python -m pytest -q tests`; accelerator tests skip when their hardware is
unavailable. Hosted Linux/macOS CI checks installation and contracts; a separate
CPU job exercises the real checkpoint. Hardware skips do not qualify MPS/CUDA.

CUDA hardware was unavailable locally. Larger checkpoints, long clips,
full notebook execution, the running web demo and training are not covered by
this workload. No custom Metal kernel, MLX port or package release is included.
