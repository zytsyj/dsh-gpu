import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: ['src/index.ts'],
  outDir: 'lib',
  dts: true,
  format: 'esm',
  target: 'node22',
  fixedExtension: false,
  exports: false,
  unbundle: true,
})
