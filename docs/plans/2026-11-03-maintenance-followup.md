# Fork maintenance follow-up

Approved scope: adapt the next three credited upstream fixes, refresh the
frontend dependencies as one coherent workstream, protect fork `main`, and add
a manual physical Apple Silicon qualification gate for future releases.
Agents Orchestrator coordinates implementation and independent verification.
PR creation and ordinary publication are authorized; PR merging, releases,
deployment and new worktrees remain outside this follow-up.

## Ordered checklist

- [x] Refresh repository state, existing PR heads, CI and review feedback.
- [x] Implement the bounded Python fixes, frontend refresh and native release
  qualification command in disjoint files.
- [x] Run focused tests, full integration checks and independent review.
- [ ] Publish focused fork PRs and verify CI (active). Explicit signed-commit
  and publication authorization is recorded; main protections are applied.
- [ ] Read back published revisions, protections and remaining limitations.

## Ownership and acceptance

The AI Engineer owns PathLike video dispatch, Hugging Face `cache_dir`
forwarding and metadata-safe training discovery, with behavioral regression
tests. Original proposals are [#475 by @smorrin28](https://github.com/facebookresearch/sam2/pull/475),
[#285 by @rickstaa](https://github.com/facebookresearch/sam2/pull/285) and
[#370 by @jayisaking](https://github.com/facebookresearch/sam2/pull/370).

The Frontend Developer owns one regenerated frontend manifest/lockfile and
necessary toolchain migration. The overlapping Dependabot PRs are #755, #754,
#752, #750, #745, #744 and #743. Acceptance requires current advisory evidence,
consistent installation, type/lint/build checks and bounded browser evidence.
Unresolved advisories must be reported separately from a passing build.

The primary agent owns workflow integration, branch protections, commits and
publication. The native qualification command requires a clean committed
checkout, a reported physical Apple M-series GPU, working MPS, fallback unset,
the complete test suite and unchanged real-checkpoint CPU/MPS quality gates.
Its receipt identifies source revision, checkpoint, environment and skipped
tests; hosted GPU skips cannot satisfy this maintainer release checklist.

## Initial live checkpoint

At the start of this follow-up, PR #1 is at `0c1a401`, PR #2 at `c93f9ba` and
PR #3 at `874f693`; all are open and unmerged. PR #3 has six passing checks.
PR #1 and #2 each have five passing checks and a cancelled current-PyTorch
macOS 15 job. Both cancelled jobs were never acquired by a hosted runner;
GitHub annotations explicitly report macOS ARM capacity constraints. These
jobs provide no coverage and are not product failures or passing gates.
PR #2 has already consumed one capacity retry. No retry is implied here.

The proposed `main` protection uses all six verified GitHub Actions contexts,
requires an up-to-date PR and resolved conversations, applies to administrators,
and prevents force pushes/deletion. Zero mandatory human approvals avoids
locking out the sole collaborator. Adding a frontend required check is deferred
until that workflow exists on `main`, avoiding a deadlock for the earlier stack.
After an automatic approval review required exact authorization, the user
explicitly approved this configuration. The six-check protection is now saved
on `main`; an independent API readback matches the approved settings. The
frontend check remains deferred until its workflow reaches `main`.

One bounded retry of PR #1's cancelled job passed on its unchanged `0c1a401`
head; all six checks are now green. PR #2 retains its earlier capacity blocker.
Automatic approval review required an explicit commit request. The user then
authorized signed commits, clean-source qualification, ordinary fork pushes and
two focused PRs. Publication follows the final clean-source checks; merges and
releases remain excluded.

All three PRs currently have no submitted GitHub reviews or inline review
threads. Earlier independent reviews remain historical evidence for their
recorded source revisions. New changes require independent review before
publication claims. No branch protection bypass or release is planned.

## Local validation checkpoint

The reviewed Python changes pass the final full suite on the physical M3 Pro
with both PyTorch 2.14.1 / TorchVision 0.29.1 and PyTorch 2.15.0 / TorchVision
0.29.2: 174 passed, six CUDA hardware skips in each environment. Native MPS
allocation, arithmetic and synchronization pass with fallback unset.
The unchanged real-checkpoint CPU/MPS gates pass on both runtimes: image IoU
1.0, automatic-mask matched fraction 1.0, minimum video IoU 0.9997682503 and
zero retained allocation growth. See the [source-hash-bound receipt](../validation/mps-maintenance-2026-11-03.json).

Independent review found and closed two defects in the new work: inherited
pytest selection could shorten release qualification, and explicit training
file lists could retain `.DS_Store`. The corrected gate has 20 focused cases;
its full positive end-to-end run still requires a clean committed checkout.

Frontend validation uses isolated Node 24.22.2 and Yarn 1.22.22. Frozen
installation, Relay generation, lint, TypeScript and production build pass.
The production audit has zero advisories; the full audit retains one moderate
lint-only advisory. Frontend independent review passes, with a separate
[validation receipt](../validation/frontend-maintenance-2026-11-03.json) recording
advisory details and bounded browser scope. This dependency refresh does not
establish a live model-backed browser session or upload/export acceptance.
