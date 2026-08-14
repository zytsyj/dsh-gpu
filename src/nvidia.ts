/**
 * nvidia-smi querying and parsing: one snapshot of every GPU on the host.
 *
 * Pure parsing lives here so tests can feed recorded output; the executor
 * injection lives in the plugin entry. Queries use CSV format for
 * machine-parseability and are resilient to unit suffixes (MiB/GiB).
 *
 * @module dsh-gpu/nvidia
 */

/** One GPU device snapshot. */
export interface GpuDevice {
  /** Physical index as in CUDA ordering. */
  index: number
  /** Device name, e.g. 'Tesla V100-SXM2-32GB'. */
  name: string
  /** Total device memory in MiB. */
  memoryTotalMiB: number
  /** Memory currently in use, MiB. */
  memoryUsedMiB: number
  /** SM utilization percent, 0-100. */
  utilizationPct: number
  /** Temperature in Celsius. */
  temperatureC: number
}

/** Whole-host snapshot. */
export interface GpuSnapshot {
  devices: GpuDevice[]
  /** Wall-clock time of the snapshot. */
  sampledAt: number
}

/** Device considered busy at or above either threshold. */
const BUSY_MEMORY_PCT = 80
const BUSY_UTIL_PCT = 50

/** A device is free when both memory and SM utilization are low. */
export function isFree(device: GpuDevice, memoryPct = BUSY_MEMORY_PCT, utilPct = BUSY_UTIL_PCT): boolean {
  const memoryUsedPct = device.memoryTotalMiB > 0
    ? (device.memoryUsedMiB / device.memoryTotalMiB) * 100
    : 100
  return memoryUsedPct < memoryPct && device.utilizationPct < utilPct
}

/** Sort key: freest first (most absolute free memory, then lowest utilization). */
export function freenessScore(device: GpuDevice): number {
  return (device.memoryTotalMiB - device.memoryUsedMiB) * 100 - device.utilizationPct
}

/** Parse one CSV row of `nvidia-smi --query-gpu`. */
export function parseQueryCsv(csv: string): GpuSnapshot {
  const lines = csv.split('\n').map(line => line.trim()).filter(line => line.length > 0)
  const header = lines[0]?.split(',').map(field => field.trim()) ?? []
  // nvidia-smi CSV headers carry unit suffixes ('memory.total [MiB]',
  // 'utilization.gpu [%]'), so match by prefix, not equality.
  const columnOf = (name: string): number =>
    header.findIndex(h => h === name || h.startsWith(`${name} `) || h.startsWith(`${name}[`))
  const indexColumn = columnOf('index')
  if (indexColumn < 0) return { devices: [], sampledAt: Date.now() }
  const devices: GpuDevice[] = []
  for (const line of lines.slice(1)) {
    const fields = line.split(',').map(field => field.trim())
    const get = (name: string): string => {
      const i = columnOf(name)
      return i >= 0 ? (fields[i] ?? '') : ''
    }
    const index = Number(get('index'))
    if (!Number.isInteger(index) || index < 0) continue
    devices.push({
      index,
      name: get('name'),
      memoryTotalMiB: parseMiB(get('memory.total')),
      memoryUsedMiB: parseMiB(get('memory.used')),
      utilizationPct: parseLeadingInt(get('utilization.gpu')),
      temperatureC: parseLeadingInt(get('temperature.gpu')),
    })
  }
  return { devices, sampledAt: Date.now() }
}

/** '24576 MiB' / '24 GB' / '24576' -> MiB integer. */
export function parseMiB(value: string): number {
  const m = value.match(/^([\d.]+)\s*([KMG]i?B)?/i)
  if (m === null) return 0
  const n = Number(m[1])
  if (!Number.isFinite(n)) return 0
  switch ((m[2] ?? '').toUpperCase()) {
    case 'KB': case 'KIB': return Math.round(n / 1024)
    case 'GB': case 'GIB': return Math.round(n * 1024)
    default: return Math.round(n)
  }
}

/** '42 %' / '42' -> 42. */
export function parseLeadingInt(value: string): number {
  const m = value.match(/^(\d+)/)
  return m === null ? 0 : Number(m[1])
}

/** One-line human summary for prompt injection. */
export function summarize(
  snapshot: GpuSnapshot,
  memoryPct = BUSY_MEMORY_PCT,
  utilPct = BUSY_UTIL_PCT,
): string {
  if (snapshot.devices.length === 0) return 'No NVIDIA GPUs detected.'
  const free = snapshot.devices.filter(d => isFree(d, memoryPct, utilPct))
  const parts = snapshot.devices.map(d =>
    `GPU${d.index} ${d.name}: ${d.memoryUsedMiB}/${d.memoryTotalMiB}MiB ${d.utilizationPct}%util`
      + (d.temperatureC > 0 ? ` ${d.temperatureC}C` : ''),
  )
  const head = `${snapshot.devices.length} GPU(s): ${free.length} free`
  return `${head} — ${parts.join(' | ')}`
}

/** The command whose output {@link parseQueryCsv} consumes. */
export const QUERY_COMMAND = [
  'nvidia-smi',
  '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu',
  '--format=csv',
].join(' ')
