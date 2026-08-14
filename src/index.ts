/**
 * dsh-gpu: GPU-aware execution layer for DeepSeek Harness.
 *
 * Registers three model-facing tools on `ctx.tools`:
 * - `gpu_status`: full snapshot of every GPU (free/busy, memory, utilization)
 * - `gpu_exec`: one-shot command with automatic card selection
 * - `gpu_run_bg`: background GPU job (ctx.jobs), card held for the job
 *
 * Plus an `agent/pre-step` listener that injects a one-line GPU summary into
 * eligible steps (mirroring the official time-context plugin pattern).
 *
 * Execution rides the mounted `ctx.shell` executor — local or any remote
 * provider (e.g. dsh-ssh) — so the GPU host can be the session's execution
 * world without this plugin knowing where it is.
 *
 * @module dsh-gpu
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { isAbsolute, resolve as resolvePath } from 'node:path'
import { defineTool, TOOL_ABORTED, type ToolRunContext } from '@deepseek-ai/dsh-tools'
import { createUserMessage, HarnessError } from '@deepseek-ai/dsh-llm'
import type { PreStepDecision } from '@deepseek-ai/dsh-agent'
import type { ShellProcess } from '@deepseek-ai/dsh-shell'
import type {} from '@deepseek-ai/dsh-jobs'
import type {} from '@deepseek-ai/dsh-system-prompt'
import { QUERY_COMMAND, parseQueryCsv, summarize, isFree, freenessScore } from './nvidia.ts'
import type { GpuSnapshot } from './nvidia.ts'

declare module '@deepseek-ai/dsh-jobs' {
  interface JobKindMap {
    gpu: 'gpu'
  }
}

/** Cordis plugin name used by loader diagnostics. */
export const name = 'gpu'

/** Required services: tool registry, shell executor, system prompt, agent registry. */
export const inject = ['tools', 'shell', 'systemPrompt', 'agents']

export interface Config {
  /** Inject the one-line GPU summary into eligible steps. Default true. */
  stepContext?: boolean
  /** Minimum spacing between injected GPU snapshots, ms. Default 60_000. */
  refreshIntervalMs?: number
  /** Query timeout for nvidia-smi, ms. Default 10_000. */
  queryTimeoutMs?: number
  /** Consider a GPU busy at or above this memory-used percent. Default 80. */
  busyMemoryPct?: number
  /** Consider a GPU busy at or above this SM utilization percent. Default 50. */
  busyUtilPct?: number
}

const MAX_TIMER_DELAY_MS = 2_147_483_647

export const Config: z<Config> = z.object({
  stepContext: z.boolean().default(true),
  refreshIntervalMs: z.number().step(1).min(0).max(MAX_TIMER_DELAY_MS).default(60_000),
  queryTimeoutMs: z.number().step(1).min(1).max(MAX_TIMER_DELAY_MS).default(10_000),
  busyMemoryPct: z.number().min(0).max(100).default(80),
  busyUtilPct: z.number().min(0).max(100).default(50),
})

/** Args for gpu_exec. */
interface ExecArgs {
  command: string
  description: string
  gpuIndex?: number
  count?: number
  workdir?: string
  timeoutMs?: number
}

/** Args for gpu_run_bg. */
interface BgArgs {
  command: string
  description: string
  gpuIndex?: number
  count?: number
  workdir?: string
}

function abortToolCall(): never {
  const error = new HarnessError('tool call aborted', TOOL_ABORTED)
  error.name = 'AbortError'
  throw error
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) abortToolCall()
}

function validateConfig(config: Config): void {
  const validateTimer = (key: 'refreshIntervalMs' | 'queryTimeoutMs', value: number | undefined, min: number): void => {
    if (value !== undefined && (!Number.isSafeInteger(value) || value < min || value > MAX_TIMER_DELAY_MS)) {
      throw new TypeError(`dsh-gpu: ${key} must be an integer from ${min} to ${MAX_TIMER_DELAY_MS}`)
    }
  }
  const validatePct = (key: 'busyMemoryPct' | 'busyUtilPct', value: number | undefined): void => {
    if (value !== undefined && (!Number.isFinite(value) || value < 0 || value > 100)) {
      throw new TypeError(`dsh-gpu: ${key} must be a number from 0 to 100`)
    }
  }
  validateTimer('refreshIntervalMs', config.refreshIntervalMs, 0)
  validateTimer('queryTimeoutMs', config.queryTimeoutMs, 1)
  validatePct('busyMemoryPct', config.busyMemoryPct)
  validatePct('busyUtilPct', config.busyUtilPct)
}

function validateCommandArgs(args: ExecArgs | BgArgs): void {
  if (args.command.trim().length === 0) {
    throw new Error('invalid command: expected a non-empty string')
  }
  if (args.description.trim().length === 0) {
    throw new Error('invalid description: expected a non-empty string')
  }
  if (args.gpuIndex !== undefined && args.count !== undefined) {
    throw new Error('gpuIndex and count are mutually exclusive')
  }
  if (args.gpuIndex !== undefined && (!Number.isSafeInteger(args.gpuIndex) || args.gpuIndex < 0)) {
    throw new Error(`invalid gpuIndex: expected a non-negative integer, got ${String(args.gpuIndex)}`)
  }
  if (args.count !== undefined && (!Number.isSafeInteger(args.count) || args.count < 1)) {
    throw new Error(`invalid count: expected a positive integer, got ${String(args.count)}`)
  }
  if ('timeoutMs' in args && args.timeoutMs !== undefined
    && (!Number.isSafeInteger(args.timeoutMs) || args.timeoutMs < 1 || args.timeoutMs > MAX_TIMER_DELAY_MS)) {
    throw new Error(`invalid timeoutMs: expected an integer from 1 to ${MAX_TIMER_DELAY_MS}`)
  }
}

/** Sample the GPU host once through the mounted shell executor. */
async function sampleGpus(
  ctx: Context,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<GpuSnapshot | undefined> {
  if (signal !== undefined) throwIfAborted(signal)
  let result
  try {
    result = await ctx.shell.run(ctx.shell.resolve({ command: QUERY_COMMAND, timeoutMs, signal }))
  } catch (error: unknown) {
    if (signal?.aborted === true) abortToolCall()
    throw error
  }
  if (result.aborted) abortToolCall()
  if (result.exitCode !== 0) return undefined
  const csv = [result.stdout.text, result.stderr.text].find(t => t.includes('index')) ?? result.stdout.text
  const snapshot = parseQueryCsv(csv)
  return snapshot.devices.length > 0 ? snapshot : undefined
}

/** Pick GPUs: explicit index, or the N freest devices. Exported for tests. */
export function pickGpus(
  snapshot: GpuSnapshot,
  args: { gpuIndex?: number; count?: number },
  busyMemoryPct: number,
  busyUtilPct: number,
): { indices: number[]; note: string } {
  if (args.gpuIndex !== undefined && args.count !== undefined) {
    throw new Error('gpuIndex and count are mutually exclusive')
  }
  if (args.gpuIndex !== undefined) {
    if (!Number.isSafeInteger(args.gpuIndex) || args.gpuIndex < 0) {
      throw new Error(`gpuIndex must be a non-negative integer, got ${String(args.gpuIndex)}`)
    }
    const device = snapshot.devices.find(d => d.index === args.gpuIndex)
    if (device === undefined) {
      throw new Error(`GPU ${args.gpuIndex} not found (host has ${snapshot.devices.length})`)
    }
    return { indices: [args.gpuIndex], note: `GPU ${args.gpuIndex} (pinned)` }
  }
  const count = args.count ?? 1
  if (!Number.isSafeInteger(count) || count < 1) throw new Error('count must be a positive integer')
  if (count > snapshot.devices.length) {
    throw new Error(`requested ${count} GPUs, host has ${snapshot.devices.length}`)
  }
  const freeDevices = snapshot.devices
    .filter(d => isFree(d, busyMemoryPct, busyUtilPct))
    .sort((a, b) => freenessScore(b) - freenessScore(a))
  if (freeDevices.length < count) {
    throw new Error(
      `only ${freeDevices.length} free GPU(s) of ${snapshot.devices.length}; `
      + 'pin gpuIndex to force, or free a device first (gpu_status shows current state)',
    )
  }
  const chosen = freeDevices.slice(0, count).map(d => d.index).sort((a, b) => a - b)
  return { indices: chosen, note: `GPU ${chosen.join(',')} (auto: freest ${count})` }
}

/** Resolve a model workdir relative to the invoking agent's session workspace. */
function commandWorkdir(workdir: string | undefined, exec: ToolRunContext): string | undefined {
  const sessionCwd = exec.agent?.session.header.cwd
  if (workdir === undefined || isAbsolute(workdir) || sessionCwd === undefined) return workdir ?? sessionCwd
  return resolvePath(sessionCwd, workdir)
}

/** JobOutcome from a settled shell process. */
function processOutcome(proc: ShellProcess): { status: 'completed' | 'killed' | 'failed'; detail?: string } {
  if (proc.status === 'killed') {
    return { status: 'killed', detail: proc.signal !== null ? `signal: ${String(proc.signal)}` : 'killed' }
  }
  if (proc.exitCode === 0) return { status: 'completed', detail: 'exit code: 0' }
  return { status: 'failed', detail: `exit code: ${proc.exitCode}` }
}

export function apply(ctx: Context, config: Config = {}): void {
  validateConfig(config)
  const stepContext = config.stepContext ?? true
  const refreshIntervalMs = config.refreshIntervalMs ?? 60_000
  const queryTimeoutMs = config.queryTimeoutMs ?? 10_000
  const busyMemoryPct = config.busyMemoryPct ?? 80
  const busyUtilPct = config.busyUtilPct ?? 50

  // Injection-side sampling: rate-limited refresh, failed-probe backoff.
  let cachedSnapshot: GpuSnapshot | undefined
  let cachedAt = 0
  let failedAt = 0
  const sampleForInjection = async (signal: AbortSignal): Promise<GpuSnapshot | undefined> => {
    const now = Date.now()
    if (now - cachedAt < refreshIntervalMs) return cachedSnapshot
    if (now - failedAt < refreshIntervalMs) return undefined
    const snapshot = await sampleGpus(ctx, queryTimeoutMs, signal)
    if (snapshot !== undefined) {
      cachedSnapshot = snapshot
      cachedAt = now
    } else {
      failedAt = now
    }
    return snapshot
  }

  ctx.systemPrompt.section({
    name: 'tool:gpu',
    order: 106,
    text: 'GPU tools: call gpu_status before GPU work; gpu_exec/gpu_run_bg select the freest card automatically '
      + '(CUDA_VISIBLE_DEVICES is set in the command environment); pass gpuIndex to pin a card explicitly.',
  })

  // ---- gpu_status ----
  ctx.tools.register(defineTool({
    name: 'gpu_status',
    description: 'Query the GPU host: every device, memory used/total, SM utilization, temperature, '
      + 'and which devices are currently free. Call this before any GPU work.',
    parameters: {},
    async execute(_args, exec) {
      const snapshot = await sampleGpus(ctx, queryTimeoutMs, exec.signal)
      if (snapshot === undefined) {
        return {
          kind: 'no-gpu' as const,
          detail: 'nvidia-smi query failed or returned no devices — the execution host may have no NVIDIA GPUs.',
        }
      }
      const devices = snapshot.devices.map(d => ({
        index: d.index,
        name: d.name,
        memoryTotalMiB: d.memoryTotalMiB,
        memoryUsedMiB: d.memoryUsedMiB,
        utilizationPct: d.utilizationPct,
        temperatureC: d.temperatureC,
        free: isFree(d, busyMemoryPct, busyUtilPct),
      }))
      return {
        kind: 'status' as const,
        devices,
        freeIndices: devices.filter(d => d.free).map(d => d.index),
      }
    },
    output: {
      schema: {
        oneOf: [
          {
            type: 'object',
            additionalProperties: false,
            properties: {
              kind: { type: 'string', required: true, const: 'no-gpu' },
              detail: { type: 'string', required: true },
            },
          },
          {
            type: 'object',
            additionalProperties: false,
            properties: {
              kind: { type: 'string', required: true, const: 'status' },
              devices: {
                type: 'array',
                required: true,
                items: {
                  type: 'object',
                  additionalProperties: false,
                  properties: {
                    index: { type: 'integer', required: true },
                    name: { type: 'string', required: true },
                    memoryTotalMiB: { type: 'integer', required: true },
                    memoryUsedMiB: { type: 'integer', required: true },
                    utilizationPct: { type: 'integer', required: true },
                    temperatureC: { type: 'integer', required: true },
                    free: { type: 'boolean', required: true },
                  },
                },
              },
              freeIndices: { type: 'array', required: true, items: { type: 'integer' } },
            },
          },
        ],
      },
      render: (_args, value) => [{
        type: 'text',
        text: value.kind === 'no-gpu'
          ? value.detail
          : `${value.devices.length} GPU(s), free: [${value.freeIndices.join(',')}]\n`
            + value.devices.map(d =>
              `GPU${d.index} ${d.name}: ${d.memoryUsedMiB}/${d.memoryTotalMiB}MiB ${d.utilizationPct}%util`
              + (d.temperatureC > 0 ? ` ${d.temperatureC}C` : '') + (d.free ? '' : ' [busy]')).join('\n'),
      }],
    },
  }))

  // ---- gpu_exec ----
  ctx.tools.register(defineTool({
    name: 'gpu_exec',
    description: 'Run a one-shot command on the GPU host with automatic card selection: '
      + 'CUDA_VISIBLE_DEVICES is set in the environment to the freest GPU (or the pinned gpuIndex), then the command runs '
      + 'through the session shell executor. Use for short GPU commands (probes, quick python, builds); '
      + 'for long training/inference runs use gpu_run_bg.',
    parameters: {
      command: { type: 'string', required: true, description: 'The command to execute.' },
      description: { type: 'string', required: true, description: 'What this command does, 5-10 words.' },
      gpuIndex: { type: 'integer', description: 'Pin one explicit GPU index instead of auto-selection.' },
      count: { type: 'integer', description: 'Number of GPUs to select when auto-selecting (default 1).' },
      workdir: { type: 'string', description: 'Working directory (session workspace default).' },
      timeoutMs: { type: 'integer', description: 'Timeout in milliseconds.' },
    },
    async execute(args: ExecArgs, exec) {
      validateCommandArgs(args)
      const snapshot = await sampleGpus(ctx, queryTimeoutMs, exec.signal)
      if (snapshot === undefined) {
        throw new Error('nvidia-smi unavailable on the execution host — cannot select a GPU')
      }
      const { indices, note } = pickGpus(snapshot, args, busyMemoryPct, busyUtilPct)
      const workdir = commandWorkdir(args.workdir, exec)
      throwIfAborted(exec.signal)
      let result
      try {
        result = await ctx.shell.run(ctx.shell.resolve({
          command: args.command,
          env: { CUDA_VISIBLE_DEVICES: indices.join(',') },
          ...workdir !== undefined ? { workdir } : {},
          ...args.timeoutMs !== undefined ? { timeoutMs: args.timeoutMs } : {},
          signal: exec.signal,
        }))
      } catch (error: unknown) {
        if (exec.signal.aborted) abortToolCall()
        throw error
      }
      if (result.aborted) abortToolCall()
      return {
        kind: 'exec' as const,
        gpus: indices.join(','),
        selection: note,
        exitCode: result.exitCode,
        stdout: result.stdout.text,
        stderr: result.stderr.text,
        truncated: result.stdout.truncated || result.stderr.truncated,
      }
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          kind: { type: 'string', required: true, const: 'exec' },
          gpus: { type: 'string', required: true },
          selection: { type: 'string', required: true },
          exitCode: { required: true, oneOf: [{ type: 'integer' }, { type: 'null' }] },
          stdout: { type: 'string', required: true },
          stderr: { type: 'string', required: true },
          truncated: { type: 'boolean', required: true },
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: `[gpus ${value.gpus} — ${value.selection}] exit ${value.exitCode}\n${value.stdout}${value.stderr}`,
      }],
    },
  }))

  // ---- gpu_run_bg ----
  ctx.tools.register(defineTool({
    name: 'gpu_run_bg',
    description: 'Start a long-running GPU job (training, inference server, benchmark) in the background: '
      + 'selects the freest GPU(s), sets CUDA_VISIBLE_DEVICES, and returns a job id immediately. '
      + 'Read output with job_output, stop with job_kill. The job survives this tool call.',
    parameters: {
      command: { type: 'string', required: true, description: 'The long-running command to start.' },
      description: { type: 'string', required: true, description: 'What this job does, 5-10 words.' },
      gpuIndex: { type: 'integer', description: 'Pin one explicit GPU index instead of auto-selection.' },
      count: { type: 'integer', description: 'Number of GPUs to select (default 1).' },
      workdir: { type: 'string', description: 'Working directory.' },
    },
    async execute(args: BgArgs, exec) {
      validateCommandArgs(args)
      const jobs = ctx.get('jobs')
      if (jobs === undefined) {
        throw new Error('background jobs unavailable: load @deepseek-ai/dsh-jobs and @deepseek-ai/dsh-tool-jobs')
      }
      const snapshot = await sampleGpus(ctx, queryTimeoutMs, exec.signal)
      if (snapshot === undefined) {
        throw new Error('nvidia-smi unavailable on the execution host — cannot select a GPU')
      }
      const { indices, note } = pickGpus(snapshot, args, busyMemoryPct, busyUtilPct)
      const workdir = commandWorkdir(args.workdir, exec)
      throwIfAborted(exec.signal)
      const id = jobs.start({
        kind: 'gpu',
        label: `${note}: ${args.command}`,
        ...exec.agent !== undefined ? { owner: exec.agent } : {},
        run: () => {
          const proc = ctx.shell.start(ctx.shell.resolve({
            command: args.command,
            env: { CUDA_VISIBLE_DEVICES: indices.join(',') },
            ...workdir !== undefined ? { workdir } : {},
          }))
          return {
            cancel: () => void proc.kill(),
            done: proc.done.then(() => processOutcome(proc)),
            readOutput: () => proc.readOutput().delta,
          }
        },
      })
      return { kind: 'background' as const, jobId: id, gpus: indices.join(','), selection: note }
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          kind: { type: 'string', required: true, const: 'background' },
          jobId: { type: 'string', required: true },
          gpus: { type: 'string', required: true },
          selection: { type: 'string', required: true },
        },
      },
      render: (_args, value) => [{
        type: 'text',
        text: `started GPU job ${value.jobId} on ${value.gpus} (${value.selection})`,
      }],
    },
  }))

  // ---- per-step GPU context injection (time-context pattern) ----
  if (stepContext) {
    ctx.on('agent/pre-step', async ({ signal }, next): Promise<PreStepDecision> => {
      const decision = await next()
      if (decision.kind === 'reject' || signal.aborted) return decision
      let snapshot: GpuSnapshot | undefined
      try {
        snapshot = await sampleForInjection(signal)
      } catch {
        return decision
      }
      if (snapshot === undefined || signal.aborted) return decision
      const text = `GPU status: ${summarize(snapshot, busyMemoryPct, busyUtilPct)}`
      return {
        kind: 'enter',
        messages: [
          ...decision.messages,
          createUserMessage({
            content: [{ type: 'text', text }],
            source: { kind: 'plugin', plugin: name, form: 'snapshot', sections: [{ name, text }] },
          }),
        ],
      }
    }, { prepend: true })
  }
}
