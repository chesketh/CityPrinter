import { useState } from 'react'
import PresetSelector from './PresetSelector'
import type { BoundingBox, JobStatus, GeocodeResult } from '../types'

interface Preset {
  id: string
  name: string
  category: string
  lat: number
  lon: number
  span: number
  desc: string
}

interface BuildPanelProps {
  bbox: BoundingBox | null
  job: JobStatus | null
  searchResult: GeocodeResult | null
  outputFormat: 'glb' | 'ply' | '3mf'
  onSearch: (query: string) => void
  onBuild: () => void
  onBboxChange: (bbox: BoundingBox | null) => void
  onSelectPreset: (preset: Preset) => void
  onFormatChange: (format: 'glb' | 'ply' | '3mf') => void
  disabled: boolean
}

export default function BuildPanel({
  bbox, job, searchResult, outputFormat, onSearch, onBuild, onBboxChange, onSelectPreset, onFormatChange, disabled,
}: BuildPanelProps) {
  const [query, setQuery] = useState('')
  const [manualOpen, setManualOpen] = useState(false)
  const [manualCoords, setManualCoords] = useState('')

  const handleSearch = () => {
    const trimmed = query.trim()
    if (trimmed) onSearch(trimmed)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch()
  }

  const parseBboxString = (s: string) => {
    const parts = s.split(',').map(p => parseFloat(p.trim()))
    if (parts.length !== 4 || parts.some(isNaN)) return null
    const [n, s2, e, w] = parts
    if (n <= s2 || e <= w) return null
    return { north: n, south: s2, east: e, west: w }
  }

  const handleApplyBbox = () => {
    const parsed = parseBboxString(manualCoords)
    if (parsed) onBboxChange(parsed)
  }

  const handleBboxKeyDown = (ev: React.KeyboardEvent) => {
    if (ev.key === 'Enter') handleApplyBbox()
  }

  const isBuilding = job && (job.status === 'queued' || job.status === 'running')
  const isComplete = job?.status === 'completed'
  const isFailed = job?.status === 'failed'
  const canBuild = bbox !== null && !isBuilding

  const manualValid = parseBboxString(manualCoords) !== null

  return (
    <div className="flex flex-col gap-2 px-4 py-3">
      {/* Row 1: Search + Preset selector */}
      <div className="flex gap-2 items-start">
        <div className="flex-1 flex gap-1.5 min-w-0">
          <input
            className="flex-1 min-w-0 px-3 py-2 rounded-md border border-ctp-surface1 bg-ctp-surface0
                       text-ctp-text text-sm outline-none
                       focus:border-ctp-blue focus:ring-1 focus:ring-ctp-blue/30
                       transition-colors duration-150 disabled:opacity-50
                       placeholder:text-ctp-overlay0"
            type="text"
            placeholder="Search location…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!!isBuilding}
          />
          <button
            className="px-3 py-2 rounded-md text-sm font-semibold whitespace-nowrap
                       transition-colors duration-150
                       bg-ctp-blue text-ctp-base hover:bg-ctp-blue/80
                       disabled:bg-ctp-surface1 disabled:text-ctp-overlay0 disabled:cursor-not-allowed"
            onClick={handleSearch}
            disabled={!query.trim() || !!isBuilding}
          >
            Go
          </button>
        </div>
        <div className="w-[45%] min-w-0">
          <PresetSelector disabled={disabled} onSelectPreset={onSelectPreset} />
        </div>
      </div>

      {/* Output format toggle */}
      <div className="flex items-center gap-2 px-0.5">
        <span className="text-[11px] text-ctp-overlay0">Output:</span>
        <div className="flex rounded-md overflow-hidden border border-ctp-surface1">
          <button
            className={`px-3 py-1 text-xs font-semibold transition-colors duration-150 ${
              outputFormat === 'glb'
                ? 'bg-ctp-blue text-ctp-base'
                : 'bg-ctp-surface0 text-ctp-subtext0 hover:bg-ctp-surface1'
            }`}
            onClick={() => onFormatChange('glb')}
            disabled={!!isBuilding}
          >
            🌐 GLB View
          </button>
          <button
            className={`px-3 py-1 text-xs font-semibold transition-colors duration-150 ${
              outputFormat === 'ply'
                ? 'bg-ctp-green text-ctp-base'
                : 'bg-ctp-surface0 text-ctp-subtext0 hover:bg-ctp-surface1'
            }`}
            onClick={() => onFormatChange('ply')}
            disabled={!!isBuilding}
          >
            🖨️ PLY
          </button>
          <button
            className={`px-3 py-1 text-xs font-semibold transition-colors duration-150 ${
              outputFormat === '3mf'
                ? 'bg-ctp-peach text-ctp-base'
                : 'bg-ctp-surface0 text-ctp-subtext0 hover:bg-ctp-surface1'
            }`}
            onClick={() => onFormatChange('3mf')}
            disabled={!!isBuilding}
          >
            🎨 3MF
          </button>
        </div>
      </div>

      {/* Search result one-liner */}
      {searchResult?.display_name && (
        <div className="text-xs text-ctp-subtext0 truncate px-0.5">
          {searchResult.display_name}
        </div>
      )}

      {/* Row 2: Build button + bbox summary */}
      <div className="flex gap-2 items-center">
        <button
          className="px-4 py-1.5 rounded-md text-sm font-semibold whitespace-nowrap
                     transition-colors duration-150
                     bg-ctp-green text-ctp-base hover:bg-ctp-green/80
                     disabled:bg-ctp-surface1 disabled:text-ctp-overlay0 disabled:cursor-not-allowed"
          onClick={onBuild}
          disabled={!canBuild}
        >
          {isBuilding ? 'Building…' : 'Build'}
        </button>
        {bbox && (
          <span className="text-[11px] font-mono text-ctp-overlay1 truncate select-all cursor-text"
                title="N, S, E, W — click to select, paste into manual coordinates">
            {bbox.north.toFixed(4)}, {bbox.south.toFixed(4)}, {bbox.east.toFixed(4)}, {bbox.west.toFixed(4)}
          </span>
        )}
      </div>

      {/* Progress bar — thin, only during build or just completed */}
      {job && (isBuilding || isComplete || isFailed) && (
        <div className="flex flex-col gap-1">
          {(isBuilding || isComplete) && (
            <div className="h-1.5 bg-ctp-surface1 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-[width] duration-500 ease-out
                           bg-gradient-to-r from-ctp-blue via-ctp-lavender to-ctp-blue
                           bg-[length:200%_100%] animate-shimmer"
                style={{ width: `${job.progress}%` }}
              />
            </div>
          )}
          {isBuilding && (
            <span className="text-xs text-ctp-subtext0">{job.message || 'Building…'}</span>
          )}
          {isComplete && (
            <span className="text-xs text-ctp-green font-semibold">Build complete!</span>
          )}
          {isFailed && (
            <span className="text-xs text-ctp-red">Failed: {job.message || 'Unknown error'}</span>
          )}
        </div>
      )}

      {/* Manual coordinates toggle */}
      <button
        className="flex items-center gap-1 text-[11px] text-ctp-overlay0 hover:text-ctp-subtext1
                   transition-colors duration-150 self-start"
        onClick={() => setManualOpen(!manualOpen)}
      >
        <span className={`inline-block transition-transform duration-150 ${manualOpen ? 'rotate-90' : ''}`}>
          ▸
        </span>
        Manual coordinates
      </button>

      {manualOpen && (
        <div className="animate-slide-down flex gap-1.5 items-center">
          <input className="manual-bbox-input flex-1"
            type="text"
            placeholder="N, S, E, W"
            value={manualCoords}
            onChange={(e) => setManualCoords(e.target.value)}
            onKeyDown={handleBboxKeyDown}
            disabled={!!isBuilding}
          />
          <button
            className="px-3 py-1 rounded text-xs font-semibold whitespace-nowrap
                       transition-colors duration-150
                       bg-ctp-blue text-ctp-base hover:bg-ctp-blue/80
                       disabled:bg-ctp-surface1 disabled:text-ctp-overlay0 disabled:cursor-not-allowed"
            onClick={handleApplyBbox}
            disabled={!manualValid || !!isBuilding}
          >
            Apply
          </button>
        </div>
      )}
    </div>
  )
}
