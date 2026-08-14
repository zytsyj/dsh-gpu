# dsh-gpu

GPU-aware execution layer for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (dsh). Out-of-tree plugin; no harness patches required.

Agents get three tools — `gpu_status`, `gpu_exec`, `gpu_run_bg` — plus an optional per-step GPU context line. Cards are selected automatically (freest first) with `CUDA_VISIBLE_DEVICES` set for you; pin a card explicitly when you care.

```
8 GPU(s), free: [0,1,2,3,4,5,6,7]
GPU0 Tesla V100-SXM2-32GB: 4264/32768MiB 0%util 40C
...
[gpus 1 — GPU 1 (auto: freest 1)] exit 0
```

## How it works

- **`gpu_status`** — one query, every device: memory used/total, SM utilization, temperature, and a free/busy verdict. A device is *busy* above 80% memory used or 50% utilization (both configurable).
- **`gpu_exec`** — one-shot command with a selected card: `CUDA_VISIBLE_DEVICES=<freest>` is prefixed, then the command runs through the mounted `ctx.shell` executor. Auto-select or pin `gpuIndex`; reserve `count` cards for multi-GPU commands.
- **`gpu_run_bg`** — long-running GPU jobs (training, inference servers, benchmarks) register as a `gpu` job in `ctx.jobs`: returns a job id immediately, read with `job_output`, stop with `job_kill`.
- **Per-step context** (optional, on by default) — injects a one-line GPU snapshot into eligible steps (the `time-context` pattern), rate-limited to one sample per minute.

All execution rides the **mounted shell executor**. Local host, or any remote execution world (e.g. an SSH provider plugin) — dsh-gpu doesn't know or care where the GPUs are; it queries and launches through the same seam the `bash` tool uses.

## Install

dsh-gpu is an out-of-tree plugin consumed by a profile. In your profile directory (`~/.dsh/profiles/<name>/`):

```bash
pnpm add dsh-gpu
```

Then register it in `cordis.patch.yml`:

```yaml
- insert:
    - id: gpu
      name: dsh-gpu
```

Load order note: place it after your execution-world plugins (e.g. an SSH provider) so the shell seam it queries is the one you intend.

## Configuration

```yaml
- id: gpu
  name: dsh-gpu
  config:
    stepContext: true      # per-step GPU snapshot line (default true)
    refreshIntervalMs: 60000  # min spacing between injected snapshots
    queryTimeoutMs: 10000     # nvidia-smi timeout
    busyMemoryPct: 80         # >= this % memory used => busy
    busyUtilPct: 50           # >= this % SM util => busy
```

## Notes & gotchas

- `nvidia-smi` **ignores** `CUDA_VISIBLE_DEVICES` — it always reports physical indices. `gpu_exec` selection still works as intended for CUDA programs; just don't use nvidia-smi output inside `gpu_exec` to verify the pinning.
- Selection is advisory, not a reservation: two concurrent agents can still pick the same card. For exclusive claims, pin `gpuIndex` from a `gpu_status` read in the same step.
- `gpu_run_bg` requires the jobs service in the composition (`@deepseek-ai/dsh-jobs` + `@deepseek-ai/dsh-tool-jobs`), the same dependency background `bash` has.
- Hosts without NVIDIA GPUs: `gpu_status` reports a clean `no-gpu` result instead of failing.

## Development

```bash
pnpm install
pnpm typecheck   # tsc --noEmit
pnpm test        # vitest run (15 tests, recorded V100 fixtures)
pnpm build       # tsdown -> lib/
node tests/live-v100.mjs   # optional live probe (edit SSH target first)
```

Test fixtures are recorded from a live 8× Tesla V100-SXM2-32GB host (including one occupied card) — no mocking of nvidia-smi output formats.

## License

MIT
