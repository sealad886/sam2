# Apple Silicon inference delivery

User-approved direction: retain PyTorch and existing checkpoints, make MPS a
supported inference backend, and contribute the change through a personal fork.
Training, native MLX, custom Metal kernels, and deployment are outside scope.

## Design and acceptance

- Installation skips the optional CUDA extension on Darwin unless explicitly
  overridden. `SAM2_BUILD_CUDA` and `SAM2_BUILD_ALLOW_ERRORS` retain their meanings.
  Disabling CUDA must avoid importing its build tools, including strict builds.
- Canonical builders and public frame loaders select CUDA, then MPS, then CPU
  when no device is supplied. Explicit devices remain authoritative. HF loaders
  delegate to the same builders. Demo and VOS CLI share device selection.
- MPS uses float32 inference, including stored video memory. CUDA autocast and
  compilation remain CUDA-specific; incompatible optimized requests fail early.
- Connected components retain 8-connectivity, batch isolation, background zero,
  component-area maps, int32 outputs, and the input device. Use the CUDA extension
  when available; use SciPy CPU labeling otherwise. Labels need not have the same
  numeric IDs across implementations. SciPy is justified by efficient labeling
  of many dense masks without a Python per-pixel traversal.
- Probe RoPE on actual MPS. If complex tensors fail, rotate real/imaginary pairs
  with real arithmetic on MPS and compare with the existing complex CPU reference.
- Normalize JPEG frames to float32 before any async device transfer. Default MPS
  video frames to CPU storage, retaining explicit user choices.
- Provide automated contracts and a repeatable real-checkpoint qualification
  covering image prompts, automatic masks, two-object video, reset, new sessions,
  synchronized timings, and live MPS memory. No blanket CPU operator fallback.

Alternatives considered: device selection alone cannot restore mask cleanup or
qualify video; an MLX port would replace the framework and checkpoint path. The
bounded MPS implementation addresses the requested inference workflows directly.

## Ordered work

1. Inventory upstream revision, runtime and sample inputs; establish real MPS
   failures. Owner: primary agent; Project Shepherd coordinates acceptance.
2. Add failing behavioral tests, then implement packaging and device selection
   (`setup.py`, `sam2/build_sam.py`, `sam2/utils/device.py`, public frame loaders,
   demo predictor and `tools/vos_inference.py`). Risk: medium; follows discovery.
3. Add cleanup and video regressions, then implement CPU components and confirmed
   MPS fixes (`sam2/utils/misc.py`, position encoding/attention and video predictor).
   Risk: medium; follows measured backend probes.
4. Add portable qualification tooling and CPU/macOS CI; update README, INSTALL,
   and contributor test guidance. Risk: low; follows implementation.
5. Run focused tests, strict install/wheel checks, formatting, and real MPS/CPU
   comparisons. Obtain independent Python/code review and resolve findings.
6. Prepare a coherent commit and PR description, fork into the authenticated
   personal namespace, publish the task branch and open the upstream PR. Record
   exact URLs, revisions, CI and any administrative acceptance gaps. No merge,
   tag, package publication, or deployment is authorized.

## Prospective qualification gates

Use the official SAM 2.1 tiny checkpoint, existing truck image, and first three
bedroom JPEG frames at the model's native 1024 input resolution. Compare MPS with
CPU float32 using identical prompts/settings and checkpoint. Require finite
outputs, correct shapes, nonempty masks, image binary-mask IoU >= 0.99, each
object/frame video IoU >= 0.98, and at least 90% of CPU automatic masks matched
to an MPS mask at IoU >= 0.95. These are engineering regression gates, not an
accuracy benchmark or a claim about arbitrary datasets. RoPE arithmetic must
agree within float32 tolerance (absolute/relative 1e-5).

Run three MPS video sessions after warmup. After releasing session tensors and
synchronizing, live allocation must not grow by more than 16 MiB from the first
cleaned session. Report driver cache separately. Report actual hardware, OS,
Python/PyTorch versions, checkpoint identity, prompts, settings, timing samples
and memory. CUDA hardware qualification is unavailable unless discovered locally;
mocked dispatch checks do not replace it. MP4 decoding is an optional dependency
and must be reported separately from core JPEG video qualification.

## Lifecycle

Medium feature change, medium risk. Context, acceptance, design and planning are
recorded here; implementation -> quality -> documentation -> independent review
-> release preparation follow. Coordination applies to fork/PR operations.
No separate security or infrastructure project is needed for this local
inference change. Retrospective is future work pending a meaningful user outcome.

## Status

- [x] Inspect clean checkout and current upstream revision.
- [x] Probe MPS/runtime and install project environment.
- [x] Implement tested portability and cleanup.
- [x] Qualify real workflows and document evidence.
- [x] Independent Python/code and CI review; findings corrected and re-reviewed.
- [x] Final wheel build and isolated installed-wheel checkpoint/config/cleanup checks.
- [x] Publish [personal fork](https://github.com/sealad886/sam2) branch
  `codex/apple-silicon-inference` and [upstream PR #778](https://github.com/facebookresearch/sam2/pull/778).

Implementation commit: `d1a444261286860d565211c27f903a8baed86dbe` (GPG signature
verified). Local implementation, qualification and contribution preparation are
complete. Upstream acceptance remains external: Meta CLA Check reports
`ACTION_REQUIRED`, maintainer review is required, and both Actions workflows
reported `action_required` before any jobs ran. The contributor must complete
the Meta CLA personally; upstream maintainers control review and CI approval.

Runtime investigation added two confirmed Hiera fixes: MPS query pooling needs a
contiguous input, and unsupported bicubic interpolation needs a targeted CPU
positional-grid path. Optional MP4 decoding also changed to PyAV because the
former decoder's wheels could not resolve on Apple Silicon. The public input
format remains MP4 paths or bytes; no stored-data conversion is needed.

Measured gates and environment details are recorded in
[Apple Silicon qualification](../apple-silicon.md). Publication state is tracked
by the fork branch and upstream PR rather than treated as upstream acceptance.
