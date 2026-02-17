export interface BoundingBox {
  north: number
  south: number
  east: number
  west: number
}

export interface PrintLayer {
  layer: string
  file: string
  color_rgb: [number, number, number]
  faces: number
  vertices: number
  watertight: boolean
  size_mb: number
  order: number
  description: string
}

export interface JobStatus {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  message: string
  result?: {
    city_id: number
    format: 'glb' | 'ply'
    model_url?: string
    layers?: PrintLayer[]
    manifest_url?: string
    output_dir?: string
  }
  bbox?: BoundingBox
}

export interface GeocodeResult {
  north: number
  south: number
  east: number
  west: number
  display_name?: string
}
