import { useState, useEffect, useCallback, useRef } from 'react'
import Layout from './components/Layout'
import BuildPanel from './components/BuildPanel'
import MapSelector from './components/MapSelector'
import ModelViewer from './components/ModelViewer'
import PrintLayerViewer from './components/PrintLayerViewer'
import { geocodeLocation, startBuildByBbox, getJobStatus } from './api/client'
import type { BoundingBox, JobStatus, GeocodeResult } from './types'

export default function App() {
  const [bbox, setBbox] = useState<BoundingBox | null>(null)
  const [job, setJob] = useState<JobStatus | null>(null)
  const [modelUrl, setModelUrl] = useState<string | null>(null)
  const [searchResult, setSearchResult] = useState<GeocodeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [outputFormat, setOutputFormat] = useState<'glb' | 'ply'>('glb')

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [])

  const handleSearch = useCallback(async (query: string) => {
    setError(null)
    try {
      const result = await geocodeLocation(query)
      setSearchResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Geocode failed')
    }
  }, [])

  const handleBboxChange = useCallback((newBbox: BoundingBox | null) => {
    setBbox(newBbox)
  }, [])

  const startPolling = useCallback((jobId: string) => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
    }

    pollingRef.current = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId)
        setJob(status)

        if (status.status === 'completed') {
          if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
          if (status.result?.model_url) {
            setModelUrl(status.result.model_url)
          }
          // PLY results are stored in the job status itself
        } else if (status.status === 'failed') {
          if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
        }
      } catch (err) {
        console.error('Polling error:', err)
      }
    }, 1000)
  }, [])

  const handleBuild = useCallback(async () => {
    if (!bbox) return

    setError(null)
    setModelUrl(null)

    try {
      const { job_id, bbox: actualBbox } = await startBuildByBbox(bbox, outputFormat)
      if (actualBbox) {
        setBbox(actualBbox)
      }
      setJob({
        job_id,
        status: 'queued',
        progress: 0,
        message: 'Build queued...',
      })
      startPolling(job_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Build request failed')
      setJob({
        job_id: '',
        status: 'failed',
        progress: 0,
        message: err instanceof Error ? err.message : 'Build request failed',
      })
    }
  }, [bbox, outputFormat, startPolling])

  const handlePresetSelect = useCallback(async (preset: { id: string; lat: number; lon: number; span: number }) => {
    setError(null)

    // Check if pre-rendered GLB exists
    try {
      const res = await fetch(`/output/${preset.id}.glb`, { method: 'HEAD' })
      if (res.ok) {
        setModelUrl(`/output/${preset.id}.glb`)
        setJob(null)
        const cosLat = Math.cos((preset.lat * Math.PI) / 180)
        const halfSpan = preset.span / 2
        const newBbox: BoundingBox = {
          north: preset.lat + halfSpan,
          south: preset.lat - halfSpan,
          east: preset.lon + halfSpan / cosLat,
          west: preset.lon - halfSpan / cosLat,
        }
        setBbox(newBbox)
        return
      }
    } catch {
      // HEAD failed — fall through to build
    }

    // No pre-rendered file — compute bbox and build
    const cosLat = Math.cos((preset.lat * Math.PI) / 180)
    const halfSpan = preset.span / 2
    const newBbox: BoundingBox = {
      north: preset.lat + halfSpan,
      south: preset.lat - halfSpan,
      east: preset.lon + halfSpan / cosLat,
      west: preset.lon - halfSpan / cosLat,
    }
    setBbox(newBbox)
    setModelUrl(null)

    try {
      const { job_id, bbox: actualBbox } = await startBuildByBbox(newBbox, outputFormat)
      if (actualBbox) {
        setBbox(actualBbox)
      }
      setJob({
        job_id,
        status: 'queued',
        progress: 0,
        message: 'Build queued...',
      })
      startPolling(job_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Build request failed')
      setJob({
        job_id: '',
        status: 'failed',
        progress: 0,
        message: err instanceof Error ? err.message : 'Build request failed',
      })
    }
  }, [outputFormat, startPolling])

  const isBuilding = job && (job.status === 'queued' || job.status === 'running')

  const leftPanel = (
    <>
      <BuildPanel
        bbox={bbox}
        job={job}
        searchResult={searchResult}
        outputFormat={outputFormat}
        onSearch={handleSearch}
        onBuild={handleBuild}
        onBboxChange={handleBboxChange}
        onSelectPreset={handlePresetSelect}
        onFormatChange={setOutputFormat}
        disabled={!!isBuilding}
      />
      {error && (
        <div className="mx-4 mb-2 px-3 py-2 rounded-lg bg-ctp-red/10 border border-ctp-red/20
                        text-ctp-red text-[13px] animate-fade-in">
          {error}
        </div>
      )}
      <MapSelector fitTo={searchResult} bbox={bbox} onBboxChange={handleBboxChange} />
    </>
  )

  const isPlyResult = job?.status === 'completed' && job.result?.format === 'ply'

  const rightPanel = isPlyResult ? (
    <div className="flex flex-col items-center justify-center gap-5 p-10 text-center">
      <div className="text-5xl">🖨️</div>
      <div>
        <div className="text-lg font-bold text-ctp-text mb-1">Print-Ready PLY Generated</div>
        <div className="text-sm text-ctp-subtext0 mb-4">
          Single watertight mesh · {job!.result!.faces?.toLocaleString()} faces ·{' '}
          {job!.result!.size_mb} MB ·{' '}
          <span className={job!.result!.watertight ? 'text-ctp-green' : 'text-ctp-yellow'}>
            {job!.result!.watertight ? '✓ watertight' : '⚠ non-manifold'}
          </span>
        </div>
        <a
          href={job!.result!.model_url}
          download
          className="px-6 py-2.5 rounded-lg text-sm font-bold
                     bg-ctp-green text-ctp-base hover:bg-ctp-green/80
                     transition-colors duration-150"
        >
          ↓ Download PLY
        </a>
      </div>
    </div>
  ) : modelUrl ? (
    <ModelViewer url={modelUrl} />
  ) : (
    <div className="flex flex-col items-center justify-center text-ctp-overlay0 gap-4 p-10 text-center">
      <svg
        width="80"
        height="80"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-ctp-surface1"
      >
        <path d="M2 20h20" />
        <path d="M5 20V8l5-4 5 4v12" />
        <path d="M10 20v-6h0" />
        <path d="M15 20V12l5-2v10" />
        <path d="M8 12h2" />
        <path d="M8 16h2" />
      </svg>
      <div>
        <div className="text-lg font-semibold mb-2">No Model Loaded</div>
        <div className="text-sm leading-relaxed">
          Search for a location or draw a bounding box on the map,<br />
          then click "Build" to generate a city model.
        </div>
      </div>
    </div>
  )

  return <Layout left={leftPanel} right={rightPanel} />
}
