import { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, FeatureGroup, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import 'leaflet-draw/dist/leaflet.draw.css'

// Polyfill: leaflet-draw references a global `type` variable that doesn't exist
// in strict mode / modern bundlers. This is a known bug in leaflet-draw.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
;(window as any).type = ''

import 'leaflet-draw'
import type { BoundingBox } from '../types'

interface DrawControlProps {
  onBboxChange: (bbox: BoundingBox | null) => void
  hasRectRef: React.MutableRefObject<boolean>
  drawnItemsRef: React.MutableRefObject<L.FeatureGroup>
  bboxFromMapRef: React.MutableRefObject<boolean>
}

function DrawControl({ onBboxChange, hasRectRef, drawnItemsRef, bboxFromMapRef }: DrawControlProps) {
  const map = useMap()

  useEffect(() => {
    const drawnItems = drawnItemsRef.current
    map.addLayer(drawnItems)

    const drawControl = new L.Control.Draw({
      draw: {
        rectangle: {
          shapeOptions: {
            color: '#89b4fa',
            weight: 2,
            fillOpacity: 0.15,
          },
        } as L.DrawOptions.RectangleOptions,
        polyline: false,
        polygon: false,
        circle: false,
        marker: false,
        circlemarker: false,
      },
      edit: {
        featureGroup: drawnItems,
        remove: true,
      },
    })

    map.addControl(drawControl)

    const onCreated = (e: L.DrawEvents.Created) => {
      // Clear previous rectangles
      drawnItems.clearLayers()
      const layer = e.layer as L.Rectangle
      drawnItems.addLayer(layer)
      hasRectRef.current = true
      bboxFromMapRef.current = true

      const bounds = layer.getBounds()
      onBboxChange({
        north: bounds.getNorth(),
        south: bounds.getSouth(),
        east: bounds.getEast(),
        west: bounds.getWest(),
      })
    }

    const onEdited = (e: L.DrawEvents.Edited) => {
      const layers = e.layers
      layers.eachLayer((layer) => {
        const bounds = (layer as L.Rectangle).getBounds()
        onBboxChange({
          north: bounds.getNorth(),
          south: bounds.getSouth(),
          east: bounds.getEast(),
          west: bounds.getWest(),
        })
      })
    }

    const onDeleted = () => {
      if (drawnItems.getLayers().length === 0) {
        hasRectRef.current = false
        // Revert to viewport bbox
        const b = map.getBounds()
        onBboxChange({
          north: b.getNorth(),
          south: b.getSouth(),
          east: b.getEast(),
          west: b.getWest(),
        })
      }
    }

    map.on(L.Draw.Event.CREATED, onCreated as L.LeafletEventHandlerFn)
    map.on(L.Draw.Event.EDITED, onEdited as L.LeafletEventHandlerFn)
    map.on(L.Draw.Event.DELETED, onDeleted as L.LeafletEventHandlerFn)

    return () => {
      map.removeControl(drawControl)
      map.removeLayer(drawnItems)
      map.off(L.Draw.Event.CREATED, onCreated as L.LeafletEventHandlerFn)
      map.off(L.Draw.Event.EDITED, onEdited as L.LeafletEventHandlerFn)
      map.off(L.Draw.Event.DELETED, onDeleted as L.LeafletEventHandlerFn)
    }
  }, [map, onBboxChange, hasRectRef])

  return null
}

interface ViewportTrackerProps {
  onBboxChange: (bbox: BoundingBox) => void
  hasRectRef: React.MutableRefObject<boolean>
}

function ViewportTracker({ onBboxChange, hasRectRef }: ViewportTrackerProps) {
  const map = useMap()

  useEffect(() => {
    const update = () => {
      if (hasRectRef.current) return  // Don't override drawn rectangle
      const b = map.getBounds()
      onBboxChange({
        north: b.getNorth(),
        south: b.getSouth(),
        east: b.getEast(),
        west: b.getWest(),
      })
    }

    // Emit initial viewport bbox so the UI has a bbox from the start
    update()

    map.on('moveend', update)
    return () => { map.off('moveend', update) }
  }, [map, onBboxChange, hasRectRef])

  return null
}

interface FitBoundsProps {
  bbox: BoundingBox | null
}

function FitBounds({ bbox }: FitBoundsProps) {
  const map = useMap()

  useEffect(() => {
    if (bbox) {
      const bounds = L.latLngBounds(
        L.latLng(bbox.south, bbox.west),
        L.latLng(bbox.north, bbox.east)
      )
      map.fitBounds(bounds, { padding: [30, 30] })
    }
  }, [map, bbox])

  return null
}

/** Show / update a blue rectangle on the map when bbox changes externally. */
function ExternalBboxRect({
  bbox,
  drawnItemsRef,
  hasRectRef,
  bboxFromMapRef,
}: {
  bbox: BoundingBox | null
  drawnItemsRef: React.MutableRefObject<L.FeatureGroup>
  hasRectRef: React.MutableRefObject<boolean>
  bboxFromMapRef: React.MutableRefObject<boolean>
}) {
  const map = useMap()

  useEffect(() => {
    if (!bbox) return
    // Skip if the bbox was just set by the draw control (avoid redundant redraw)
    if (bboxFromMapRef.current) {
      bboxFromMapRef.current = false
      return
    }
    const bounds = L.latLngBounds(
      L.latLng(bbox.south, bbox.west),
      L.latLng(bbox.north, bbox.east),
    )
    // Replace any existing rectangle with one matching the new bbox
    drawnItemsRef.current.clearLayers()
    const rect = L.rectangle(bounds, {
      color: '#89b4fa',
      weight: 2,
      fillOpacity: 0.15,
    })
    drawnItemsRef.current.addLayer(rect)
    hasRectRef.current = true
    map.fitBounds(bounds, { padding: [30, 30] })
  }, [bbox, map, drawnItemsRef, hasRectRef, bboxFromMapRef])

  return null
}

interface MapSelectorProps {
  fitTo: BoundingBox | null
  bbox: BoundingBox | null
  onBboxChange: (bbox: BoundingBox | null) => void
}

export default function MapSelector({ fitTo, bbox, onBboxChange }: MapSelectorProps) {
  const hasRectRef = useRef(false)
  const drawnItemsRef = useRef<L.FeatureGroup>(new L.FeatureGroup())
  // Skip ExternalBboxRect when the bbox was just set by the draw control
  const bboxFromMapRef = useRef(false)

  return (
    <div className="flex-1 min-h-0">
      <MapContainer
        center={[20, 0]}
        zoom={2}
        className="w-full h-full"
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FeatureGroup>
          <DrawControl onBboxChange={onBboxChange} hasRectRef={hasRectRef} drawnItemsRef={drawnItemsRef} bboxFromMapRef={bboxFromMapRef} />
        </FeatureGroup>
        <FitBounds bbox={fitTo} />
        <ExternalBboxRect bbox={bbox} drawnItemsRef={drawnItemsRef} hasRectRef={hasRectRef} bboxFromMapRef={bboxFromMapRef} />
        <ViewportTracker onBboxChange={onBboxChange} hasRectRef={hasRectRef} />
      </MapContainer>
    </div>
  )
}
