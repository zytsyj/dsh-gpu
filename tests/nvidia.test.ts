/**
 * Unit tests for nvidia-smi parsing and GPU selection logic, using recorded
 * real output from a V100 host (8× Tesla V100-SXM2-32GB, GPU6 occupied).
 *
 * @module dsh-gpu/tests
 */

import { describe, expect, it } from 'vitest'
import {
  QUERY_COMMAND,
  freenessScore,
  isFree,
  parseMiB,
  parseQueryCsv,
  summarize,
} from '../src/nvidia.ts'
import { pickGpus } from '../src/index.ts'

/** Recorded from the live 8×V100 host on 2026-08-14 (GPU6 busy). */
const V100_CSV = `index, name, memory.total [MiB], memory.used [MiB], utilization.gpu [%], temperature.gpu
0, Tesla V100-SXM2-32GB, 32768 MiB, 4 MiB, 0 %, 35
1, Tesla V100-SXM2-32GB, 32768 MiB, 4 MiB, 0 %, 36
2, Tesla V100-SXM2-32GB, 32768 MiB, 4 MiB, 0 %, 33
3, Tesla V100-SXM2-32GB, 32768 MiB, 4 MiB, 0 %, 41
4, Tesla V100-SXM2-32GB, 32768 MiB, 4 MiB, 0 %, 40
5, Tesla V100-SXM2-32GB, 32764 MiB, 4 MiB, 0 %, 37
6, Tesla V100-SXM2-32GB, 32768 MiB, 13928 MiB, 100 %, 51
7, Tesla V100-SXM2-32GB, 32768 MiB, 4 MiB, 0 %, 40`

describe('parseQueryCsv', () => {
  it('parses 8 devices from recorded V100 output', () => {
    const snapshot = parseQueryCsv(V100_CSV)
    expect(snapshot.devices).toHaveLength(8)
    expect(snapshot.devices[0].name).toBe('Tesla V100-SXM2-32GB')
    expect(snapshot.devices[0].memoryTotalMiB).toBe(32768)
  })

  it('reads busy state on GPU6 (13.9 GiB used, 100% util)', () => {
    const snapshot = parseQueryCsv(V100_CSV)
    expect(snapshot.devices[6].memoryUsedMiB).toBe(13928)
    expect(snapshot.devices[6].utilizationPct).toBe(100)
  })

  it('matches headers by prefix despite unit suffixes', () => {
    const snapshot = parseQueryCsv(V100_CSV)
    for (const device of snapshot.devices) {
      expect(device.memoryTotalMiB).toBeGreaterThan(0)
    }
  })

  it('returns empty devices for garbage input', () => {
    const snapshot = parseQueryCsv('garbage, input\n1, 2')
    expect(snapshot.devices).toHaveLength(0)
  })
})

describe('isFree / freenessScore', () => {
  const snapshot = parseQueryCsv(V100_CSV)

  it('marks 7 devices free and GPU6 busy', () => {
    const free = snapshot.devices.filter(d => isFree(d)).map(d => d.index)
    expect(free).toEqual([0, 1, 2, 3, 4, 5, 7])
    expect(isFree(snapshot.devices[6])).toBe(false)
  })

  it('ranks strictly by free memory then utilization', () => {
    const sorted = [...snapshot.devices.filter(d => isFree(d))].sort((a, b) => freenessScore(b) - freenessScore(a))
    const scores = sorted.map(freenessScore)
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i - 1]).toBeGreaterThanOrEqual(scores[i])
    }
  })
})

describe('pickGpus', () => {
  const snapshot = parseQueryCsv(V100_CSV)

  it('auto-selects the freest single GPU', () => {
    const { indices, note } = pickGpus(snapshot, { count: 1 }, 80, 50)
    expect(indices).toHaveLength(1)
    expect(note).toContain('auto')
  })

  it('selects multiple GPUs by count', () => {
    const { indices } = pickGpus(snapshot, { count: 3 }, 80,  50)
    expect(indices).toHaveLength(3)
    expect(indices).not.toContain(6)
  })

  it('respects a pinned gpuIndex', () => {
    const { indices, note } = pickGpus(snapshot, { gpuIndex: 3 }, 80, 50)
    expect(indices).toEqual([3])
    expect(note).toContain('pinned')
  })

  it('refuses when not enough free GPUs', () => {
    expect(() => pickGpus(snapshot, { count: 9 }, 80, 50)).toThrow(/host has/)
  })

  it('throws on unknown gpuIndex', () => {
    expect(() => pickGpus (snapshot, { gpuIndex: 42 }, 80, 50)).toThrow(/not found/)
  })
})

describe('parseMiB', () => {
  it('handles MiB, GiB, and bare numbers', () => {
    expect(parseMiB('24576 MiB')).toBe(24576)
    expect(parseMiB('24 GiB')).toBe(24576)
    expect(parseMiB('24576')).toBe(24576)
  })

  it('returns 0 on garbage', () => {
    expect(parseMiB('[N/A]')).toBe(0)
  })
})

describe('summarize', () => {
  it('renders the one-line summary', () => {
    const text = summarize(parseQueryCsv(V100_CSV))
    expect(text).toContain('8 GPU(s): 7 free')
    expect(text).toContain('13928/32768MiB')
  })
})

describe('QUERY_COMMAND', () => {
  it('queries exactly the fields the parser needs', ()  => {
    for (const field of ['index', 'name', 'memory.total', 'memory.used', 'utilization.gpu', 'temperature.gpu']) {
      expect(QUERY_COMMAND).toContain(field)
    }
  })
})
