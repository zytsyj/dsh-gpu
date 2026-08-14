import type { Context } from '@deepseek-ai/cordis'
import { TOOL_ABORTED, type ToolDefinition, type ToolRunContext } from '@deepseek-ai/dsh-tools'
import { HarnessError } from '@deepseek-ai/dsh-llm'
import { describe, expect, it, vi } from 'vitest'
import { apply } from '../src/index.ts'

const GPU_CSV = `index, name, memory.total [MiB], memory.used [MiB], utilization.gpu [%], temperature.gpu
0, Test GPU, 1000 MiB, 100 MiB, 60 %, 40`

function shellResult(overrides: Record<string, unknown> = {}) {
  return {
    exitCode: 0,
    signal: null,
    timedOut: false,
    aborted: false,
    timeoutMs: 10_000,
    stdout: { text: GPU_CSV, truncated: false },
    stderr: { text: '', truncated: false },
    ...overrides,
  }
}

function toolExec(signal = new AbortController().signal, cwd?: string): ToolRunContext {
  return {
    signal,
    deferContext() {},
    concludeTurn() {},
    ...cwd !== undefined ? { agent: { session: { header: { cwd } } } } : {},
  } as unknown as ToolRunContext
}

function setup(options: {
  run?: (request: Record<string, unknown>) => Promise<ReturnType<typeof shellResult>>
  jobs?: { start: ReturnType<typeof vi.fn> }
  config?: Parameters<typeof apply>[1]
} = {}) {
  const tools = new Map<string, ToolDefinition>()
  let preStep: ((payload: { signal: AbortSignal }, next: () => Promise<unknown>) => Promise<unknown>) | undefined
  const resolve = vi.fn((request: Record<string, unknown>) => request)
  const run = vi.fn(options.run ?? (async () => shellResult()))
  const section = vi.fn()
  const register = vi.fn((tool: ToolDefinition) => {
    tools.set(tool.name, tool)
    return () => tools.delete(tool.name)
  })
  const on = vi.fn((event: string, listener: typeof preStep) => {
    if (event === 'agent/pre-step') preStep = listener
    return () => {}
  })
  const ctx = {
    shell: { resolve, run },
    tools: { register },
    systemPrompt: { section },
    get: (key: string) => key === 'jobs' ? options.jobs : undefined,
    on,
  } as unknown as Context
  apply(ctx, options.config ?? {})
  return { tools, preStep: () => preStep, resolve, run, section }
}

describe('plugin lifecycle', () => {
  it('registers all tools and its system-prompt section', () => {
    const { tools, section } = setup({ config: { stepContext: false } })
    expect([...tools.keys()]).toEqual(['gpu_status', 'gpu_exec', 'gpu_run_bg'])
    expect(section).toHaveBeenCalledOnce()
  })

  it('uses configured thresholds in status and injected context', async () => {
    const { tools, preStep } = setup({ config: { busyUtilPct: 70, refreshIntervalMs: 0 } })
    const status = await tools.get('gpu_status')?.execute({}, toolExec())
    expect(status).toMatchObject({ kind: 'status', freeIndices: [0] })

    const listener = preStep()
    expect(listener).toBeDefined()
    const decision = await listener?.(
      { signal: new AbortController().signal },
      async () => ({ kind: 'enter', messages: [] }),
    )
    expect(JSON.stringify(decision)).toContain('1 free')
  })

  it('forwards cancellation to GPU sampling and command execution', async () => {
    const requests: Record<string, unknown>[] = []
    const { tools } = setup({
      config: { stepContext: false },
      run: async (request) => {
        requests.push(request)
        return requests.length === 1
          ? shellResult()
          : shellResult({ stdout: { text: 'ok', truncated: false } })
      },
    })
    const controller = new AbortController()
    const result = await tools.get('gpu_exec')?.execute(
      { command: 'python train.py', description: 'run training', gpuIndex: 0, workdir: 'runs' },
      toolExec(controller.signal, '/workspace/project'),
    )
    expect(result).toMatchObject({ kind: 'exec', gpus: '0', exitCode: 0 })
    expect(requests).toHaveLength(2)
    expect(requests.every(request => request.signal === controller.signal)).toBe(true)
    expect(requests[1].command).toBe('python train.py')
    expect(requests[1].env).toEqual({ CUDA_VISIBLE_DEVICES: '0' })
    expect(requests[1].workdir).toBe('/workspace/project/runs')
  })

  it('rejects an aborted background preflight without starting a job', async () => {
    const jobs = { start: vi.fn() }
    const { tools, run } = setup({ config: { stepContext: false }, jobs })
    const controller = new AbortController()
    controller.abort()

    const promise = tools.get('gpu_run_bg')?.execute(
      { command: 'python train.py', description: 'run training' },
      toolExec(controller.signal),
    )
    await expect(promise).rejects.toMatchObject({
      name: 'AbortError',
      code: TOOL_ABORTED,
    } satisfies Partial<HarnessError>)
    expect(run).not.toHaveBeenCalled()
    expect(jobs.start).not.toHaveBeenCalled()
  })

  it('rejects invalid config and cross-field tool arguments', async () => {
    expect(() => setup({ config: { busyMemoryPct: 101 } })).toThrow(/busyMemoryPct/)
    const { tools } = setup({ config: { stepContext: false } })
    const promise = tools.get('gpu_exec')?.execute(
      { command: 'true', description: 'test selection', gpuIndex: 0, count: 1 },
      toolExec(),
    )
    await expect(promise).rejects.toThrow(/mutually exclusive/)
  })
})
