# Global PyTorch baseline upgrade

The user approved replacing the old PyTorch baseline and Hiera workarounds,
repairing affected loading/packaging paths, and validating before updating
upstream PR #778. Use the existing clean task branch and checkout; no new
worktree, release, deployment or upstream merge is authorized.

## Accepted design

- One global baseline: PyTorch >= 2.14.1 and TorchVision >= 0.29.1, the pair
  already exercised on this Mac. Python >= 3.10 remains supported by their
  installed metadata. Use native bicubic interpolation and strided pooling;
  remove both MPS-only Hiera workarounds. Keep the public backbone parity test.
- Model-weight loading explicitly uses `weights_only=True`. Generic checkpoint
  helpers expose an explicit `weights_only` policy, defaulting to safe weight
  loading; full-object loading is opt-in for trusted checkpoints. Trainer resume
  and inter-worker object transport explicitly load their own trusted full state.
  Do not disable safe model-weight loading globally.
- Update runtime/build dependency floors, matching CI pairs and the Docker
  environment together. Verify the published image/tag rather than guessing.
  Install the real package from its canonical location without duplicating a
  shadow package in the demo server directory.
- Native SAM2 checkpoints keep their current format. No stored data in this
  checkout needs conversion. Full-state loading remains explicit; no legacy
  compatibility adapter or automatic unsafe checkpoint conversion is added.
- Retain device selection, portable mask cleanup, float32 MPS video state,
  corrected JPEG input and PyAV decoding. These remain necessary backend support.

## Execution

1. **Loading policy:** `training/trainer.py`,
   `training/utils/checkpoint_utils.py`, `training/utils/distributed.py` and
   focused tests. Prove tensor-state success, default refusal of full Python
   payloads, explicit trusted NumPy/custom-state loading and trainer resume.
   Owner: loading worker. Risk: medium, serialization/trust boundary.
2. **Packaging:** `setup.py`, `pyproject.toml`, `backend.Dockerfile`,
   `.github/workflows/inference.yml` and package-contract tests if needed.
   Verify matching runtime/build floors and clean installs. Owner: packaging
   worker. Risk: medium, native dependencies and container environment.
3. **Native Hiera:** `sam2/modeling/backbones/hieradet.py`, related tests and
   canonical docs. Remove obsolete branches; explicitly safe-load trunk weights.
   Owner: primary. Risk: medium, numerical behavior.
4. **Validation:** full CPU/MPS test suite, real tiny-checkpoint image/AMG/video
   agreement and repeated sessions without blanket fallback, real MP4 decode,
   trainer/checkpoint/CPU distributed smoke tests, strict clean install, wheel
   build/install and container/native checks where the available host permits.
   Run independently reviewed loading and packaging changes together.
5. **Contribution:** independent Python/security/CI review, resolve findings,
   rerun affected checks, inspect/sign coherent commits, push to the verified
   fork and update the PR description/evidence only after local validation.

## Validation gates

- Preserve the existing prospective image IoU >= 0.99, automatic-mask matching
  >= 90% at IoU >= 0.95, every object/frame/session video IoU >= 0.98, reset/reuse
  and <= 16 MiB retained MPS allocation growth over three sessions.
- Test Hiera with native operations and no CPU fallback environment variable.
- Round-trip tensors/primitives and explicitly trusted non-tensor checkpoint
  state; reject unsafe default model loads. Run a representative CPU optimizer
  update/resume and distributed object exchange where feasible.
- Verify native TorchVision NMS, editable install, wheel contents and import
  outside the checkout. Resolve development/demo/notebook extras coherently.
- Record CUDA extension, CUDA compilation/training and Docker checks separately.
  No CUDA hardware or running container engine is assumed. Missing hardware is
  a verification gap, never evidence that a test passed.

## Checklist

- [x] Inspect clean task branch, existing PR and accepted design.
- [x] Implement loading, packaging and native Hiera changes.
- [x] Complete local validation and refresh qualification evidence.
- [x] Independent review and corrections.
- [ ] Signed commits and verified PR update (active).

## Lifecycle and evidence

Medium feature change, medium risk, change authority. The accepted design above
supplies context, acceptance, design and planning. Implementation -> quality ->
documentation -> independent review -> contribution preparation follows.
Security review is activated for serialization trust boundaries; coordination
is activated for the existing PR update. Retrospective is future work pending
an observed user outcome. This does not authorize release or upstream merge.

Local evidence: 91 tests passed, six CUDA skips; native MPS image IoU 1.0,
automatic masks 12/12, video minimum IoU 0.999768 and zero retained allocation
growth. Native MP4 minimum IoU 0.999770. Real CPU optimizer/resume and two-rank
Gloo object transport pass. Full unfrozen tiny-model CPU forward/backward/AdamW
and canonical MOSE transforms pass. Strict fresh editable/sdist/wheel installs,
packaged model prediction/config/NMS/cleanup and formatting pass. Current
measurements and remaining CUDA/Docker gaps are in
[qualification evidence](../apple-silicon.md).

Independent review found and closed a container PEP 668 blocker: the published
base image retains externally managed Ubuntu Python. The Dockerfile now creates
a virtual environment with system-site-package access before installing the
application. All nine torch.load sites, loading callers, dependency/CI floors,
packaging sources and native Hiera were reviewed; no actionable finding remains.
