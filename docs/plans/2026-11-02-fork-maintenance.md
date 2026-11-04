# SAM 2 Apple Silicon fork maintenance

User-approved direction: maintain `sealad886/sam2` with Apple Silicon/MPS
inference through current PyTorch, preserve Meta attribution and checkpoint
formats, migrate the contribution to the fork's `main`, own its CI, and review
all open upstream pull requests for credited, deduplicated adaptations.

GitHub PRs cannot change their base repository. The contribution was recreated as
[fork PR #1](https://github.com/sealad886/sam2/pull/1), targeting the fork's
`main`; our own upstream PR #778 was closed with a replacement link. Other
contributors' PRs remain unchanged. Creating curated
fork PRs is authorized; merging them is a separate decision. No new worktree,
force push, release or deployment is authorized.

## Ordered work and gates

- [x] Establish fork identity, contribution destination and CI behavior.
- [x] Implement fork README and fork-owned Actions; independent static review passes.
- [x] Review the frozen set of all 67 open upstream PRs and record dispositions.
- [ ] Publish the implemented, tested and independently reviewed adaptations in
  focused fork PRs with original-source credit (active).
- [ ] Publish and read back fork PRs, CI outcomes and remaining limitations.

The review snapshot was fetched on 2 November 2026 from
`facebookresearch/sam2`. The local starting revision is
`ba6db50a86af940f5de9a5869946abb90f75966a`. Review evidence belongs in
[the upstream review ledger](../upstream-pr-review.md); machine-readable entries
record each PR's author, exact reviewed head, changed files, rationale,
deduplication and required verification. A recommendation is not a claim that
an adaptation has been implemented or merged.

Agents Orchestrator coordinates bounded specialists. Source reviewers read
patches and current callers; implementation workers own disjoint files; an
independent reviewer checks the integrated changes. Existing branch and user
work are preserved. MPS/CUDA hardware claims require actual hardware runs.

## Review result

All 67 entries have exact upstream heads, authors, changed files, patch and
discussion hashes, an explicit disposition, deduplication and test requirements.
Seven bounded adaptations are recommended; four are selected for the current
batch (#777, #776, #766 and #718). The next queue is #475, #285 and #370.
Seventeen PRs are already covered, three duplicate other proposals, 21 are
deferred, 11 need design/hardware qualification and eight should not be adopted
as submitted. No original upstream branch was executed or merged.

The README and two workflow changes received an independent review: all PR
bases are tested, tokens remain read-only, no pull_request_target execution is
introduced, wheel imports are checked outside the source checkout, and original
Meta research/license attribution remains. Runtime checks and provider readback
are owned by the primary agent and must be recorded before publication claims.

## Delivery checkpoint

The foundation commit `1ce370e2b5a01c34e6fcaaf247928acdabc33067` is signed and
published through fork PR #1. The fork's About description is updated. The four
selected upstream adaptations (#777, #776, #766 and #718) are implemented and
tested locally, with no open findings from independent code review; their fork
PRs have not yet been created at this checkpoint. They are not merged.

The integrated suite passed 115 tests on the physical Apple Silicon Mac, with
six CUDA hardware skips; all 12 available MPS cases executed. Eight new hardware
classification cases are included. The focused core suite passed 12 CPU predictor tests.
Four actual MP4 metadata/orientation cases passed on both PyAV 15 and PyAV 19.
These prove the primary corrected behaviors. Extra stride/reindexing scenarios,
mixed empty/nonempty crops and all combinations of point/box/mask prompts remain
follow-up coverage, not completed acceptance claims.

Five of six foundation CI jobs passed before the runner repair. The minimum
PyTorch job on hosted macOS 14 failed twice before SAM 2 execution: PyTorch
2.14 advertises the runner's unsupported Apple Paravirtual GPU, then fails its
first tiny allocation. Current PyTorch 2.15 rejects that virtual GPU explicitly.
Shared hardware preflight and migration to the macOS 15 runner were independently
reviewed with no open findings. Signed repair commit
`4f499397432323ea8e5b37ffcf03aafe5177bad4` is published; its new hosted CI results
are pending.
Hosted virtual-GPU results do not replace the physical Mac qualification.

Fresh [official tiny-checkpoint mask-batch qualification](../validation/mps-mask-batches-2026-11-02.json)
passed on native MPS with no blanket fallback. Both CPU and MPS batched masks
matched their serial predictions at IoU 1.0. Nonempty synthetic mask prompts
agreed between CPU and MPS at IoU 1.0 and 0.9999575624. The point-logit refinement
case produced empty masks on both backends; its agreement is retained as
numerical/shape evidence and is not foreground accuracy. The receipt's predictor
hash matches the reviewed implementation.
