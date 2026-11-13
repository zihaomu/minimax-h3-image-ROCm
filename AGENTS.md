# MiniMax H3-Image ROCm

Follow the parent workspace AMD environment rules and `vibe/PLAN.md`. Read
`vibe/STATE.md` before resuming work and update it with actual qualification.

- Use the independent ComfyUI + INT8 ConvRot route with our own Image compute
  and multi-GPU implementation. vLLM-Omni is not an implementation dependency.
- Keep ROCm 10.1 and the foundation's exact Torch/Triton pairing. Use the common
  workspace launcher for GPU work; isolate application dependencies and native
  build/JIT caches under this project. Rebuild extensions when ABI changes.
- Physical GPU3 is excluded. Check occupancy, shared reservations, ownership
  and physical-to-logical mapping before GPU runs. Reuse the common lock
  namespace; only clean up resources owned by the current experiment.
- The sibling `minimax_h3` is a reference. Keep its environment and historical
  artifacts intact. Record source commit/file hashes and licenses when porting.
- Establish native Image baselines before changing attention, math or transport.
  Keep those changes independently attributable. Explicit backend failures must
  not silently switch numerical paths. Video qualification does not transfer.
- Preserve Image's six-step schedule and declared workload in comparisons.
  Freeze numerical policy before evaluating candidates; retain failed evidence.
- `vibe/` and `research/` are local-only and must never enter public commits or
  their ancestor history. Keep local plans, state and source audits there.
  `models/`, `.runtime/`, `runs/`, raw tensors/traces and credentials are local.
  Publish only deliberately prepared source/configuration and public docs.
  Do not copy the sibling's protected research/archive tree into public commits.
- Sync reviewed project source/docs/configuration to the user-designated
  `zihaomu/minimax-h3-image-ROCm` repository with normal commits and pushes.
  Report support and performance only within completed validation scope.
