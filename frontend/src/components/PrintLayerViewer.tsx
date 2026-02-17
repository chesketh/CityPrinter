import type { PrintLayer } from '../types'

interface PrintLayerViewerProps {
  layers: PrintLayer[]
  outputDir: string
}

function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map(c => c.toString(16).padStart(2, '0')).join('')
}

export default function PrintLayerViewer({ layers, outputDir }: PrintLayerViewerProps) {
  const totalFaces = layers.reduce((sum, l) => sum + l.faces, 0)
  const totalSize = layers.reduce((sum, l) => sum + l.size_mb, 0)

  return (
    <div className="flex flex-col h-full bg-ctp-base">
      {/* Header */}
      <div className="px-5 py-4 border-b border-ctp-surface1">
        <h2 className="text-lg font-bold text-ctp-text flex items-center gap-2">
          🖨️ Print Layers
        </h2>
        <p className="text-xs text-ctp-subtext0 mt-1">
          {layers.length} layers · {totalFaces.toLocaleString()} faces · {totalSize.toFixed(1)} MB total
        </p>
      </div>

      {/* Layer list */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
        {layers.map((layer) => {
          const hex = rgbToHex(...layer.color_rgb)
          const downloadUrl = `${outputDir}${layer.file}`

          return (
            <div
              key={layer.layer}
              className="flex items-center gap-3 p-3 rounded-lg bg-ctp-surface0
                         border border-ctp-surface1 hover:border-ctp-blue/40
                         transition-colors duration-150"
            >
              {/* Color swatch */}
              <div
                className="w-8 h-8 rounded-md border border-ctp-surface2 flex-shrink-0"
                style={{ backgroundColor: hex }}
                title={`RGB(${layer.color_rgb.join(', ')})`}
              />

              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-ctp-text capitalize">
                    {layer.layer}
                  </span>
                  {layer.watertight ? (
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-ctp-green/15 text-ctp-green font-medium">
                      ✓ watertight
                    </span>
                  ) : (
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-ctp-yellow/15 text-ctp-yellow font-medium">
                      ⚠ non-manifold
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-ctp-subtext0 truncate">
                  {layer.description}
                </p>
                <p className="text-[10px] text-ctp-overlay0 mt-0.5">
                  {layer.faces.toLocaleString()} faces · {layer.vertices.toLocaleString()} verts · {layer.size_mb} MB
                </p>
              </div>

              {/* Download button */}
              <a
                href={downloadUrl}
                download={layer.file}
                className="flex-shrink-0 px-3 py-1.5 rounded-md text-xs font-semibold
                           bg-ctp-blue text-ctp-base hover:bg-ctp-blue/80
                           transition-colors duration-150"
              >
                ↓ PLY
              </a>
            </div>
          )
        })}
      </div>

      {/* Download all */}
      <div className="px-4 py-3 border-t border-ctp-surface1">
        <div className="flex gap-2">
          <a
            href={`${outputDir}manifest.json`}
            download="manifest.json"
            className="flex-1 text-center px-4 py-2 rounded-md text-sm font-semibold
                       bg-ctp-surface1 text-ctp-text hover:bg-ctp-surface2
                       transition-colors duration-150"
          >
            📋 Manifest
          </a>
        </div>
        <p className="text-[10px] text-ctp-overlay0 mt-2 text-center">
          Print base first → stack layers in order → alignment pegs keep registration
        </p>
      </div>
    </div>
  )
}
