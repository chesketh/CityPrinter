import type { BoundingBox, JobStatus, GeocodeResult } from '../types'

interface BuildResponse {
  job_id: string
  bbox?: BoundingBox
}

export async function geocodeLocation(query: string): Promise<GeocodeResult> {
  const response = await fetch(`/api/geocode?query=${encodeURIComponent(query)}`)
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(err.detail || 'Geocode request failed')
  }
  return response.json()
}

export async function startBuildByName(location: string): Promise<{ job_id: string }> {
  const response = await fetch('/api/build/name', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ location }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(err.detail || 'Build request failed')
  }
  return response.json()
}

export async function startBuildByBbox(
  bbox: BoundingBox,
  outputFormat: 'glb' | 'ply' = 'glb',
  scale: number = 1.0,
): Promise<BuildResponse> {
  const response = await fetch('/api/build/bbox', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...bbox, output_format: outputFormat, scale }),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(err.detail || 'Build request failed')
  }
  return response.json()
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`/api/build/status/${encodeURIComponent(jobId)}`)
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(err.detail || 'Status request failed')
  }
  return response.json()
}
