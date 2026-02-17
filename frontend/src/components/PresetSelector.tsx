import { useState, useRef, useEffect, useMemo } from 'react'
import presets from '../data/presets.json'

interface Preset {
  id: string
  name: string
  category: string
  lat: number
  lon: number
  span: number
  desc: string
}

interface PresetSelectorProps {
  disabled: boolean
  onSelectPreset: (preset: Preset) => void
}

const CATEGORIES = ['Downtown', 'Landmark', 'Nature', 'Coastal', 'Cultural', 'Urban Park'] as const

export default function PresetSelector({ disabled, onSelectPreset }: PresetSelectorProps) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [highlightIndex, setHighlightIndex] = useState(-1)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Filter presets by query
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return presets as Preset[]
    return (presets as Preset[]).filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.desc.toLowerCase().includes(q) ||
        p.category.toLowerCase().includes(q)
    )
  }, [query])

  // Group filtered presets by category
  const grouped = useMemo(() => {
    const groups: { category: string; items: Preset[] }[] = []
    for (const cat of CATEGORIES) {
      const items = filtered.filter((p) => p.category === cat)
      if (items.length > 0) {
        groups.push({ category: cat, items })
      }
    }
    return groups
  }, [filtered])

  // Flat list for keyboard navigation
  const flatItems = useMemo(() => grouped.flatMap((g) => g.items), [grouped])

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  // Scroll highlighted item into view
  useEffect(() => {
    if (highlightIndex >= 0 && dropdownRef.current) {
      const el = dropdownRef.current.querySelector(`[data-index="${highlightIndex}"]`)
      if (el) {
        el.scrollIntoView({ block: 'nearest' })
      }
    }
  }, [highlightIndex])

  const handleSelect = (preset: Preset) => {
    setQuery(preset.name)
    setOpen(false)
    setHighlightIndex(-1)
    onSelectPreset(preset)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        setOpen(true)
        setHighlightIndex(0)
        e.preventDefault()
      }
      return
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightIndex((prev) => (prev < flatItems.length - 1 ? prev + 1 : prev))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightIndex((prev) => (prev > 0 ? prev - 1 : 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (highlightIndex >= 0 && highlightIndex < flatItems.length) {
        handleSelect(flatItems[highlightIndex])
      }
    } else if (e.key === 'Escape') {
      setOpen(false)
      setHighlightIndex(-1)
    }
  }

  let flatIndex = -1

  return (
    <div className="relative" ref={wrapperRef}>
      <input
        className="w-full px-3 py-2 rounded-md border border-ctp-surface1 bg-ctp-surface0
                   text-ctp-text text-sm outline-none
                   focus:border-ctp-blue focus:ring-1 focus:ring-ctp-blue/30
                   transition-colors duration-150 disabled:opacity-50
                   placeholder:text-ctp-overlay0"
        type="text"
        placeholder="Search presets… (e.g. Manhattan, Eiffel Tower)"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value)
          setOpen(true)
          setHighlightIndex(0)
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      {open && grouped.length > 0 && (
        <div
          className="absolute top-full left-0 right-0 mt-1 max-h-80 overflow-y-auto
                     bg-ctp-surface0 border border-ctp-surface1 rounded-md z-[100]
                     shadow-xl shadow-black/30 animate-slide-down scrollbar-thin"
          ref={dropdownRef}
        >
          {grouped.map((group) => (
            <div key={group.category}>
              <div className="px-3 pt-2 pb-1 text-[10px] font-bold text-ctp-overlay0 uppercase tracking-wider">
                {group.category}
              </div>
              {group.items.map((preset) => {
                flatIndex++
                const idx = flatIndex
                const isHighlighted = idx === highlightIndex
                return (
                  <div
                    key={preset.id}
                    data-index={idx}
                    className={`flex flex-col gap-0.5 px-3 py-2 cursor-pointer transition-colors duration-75
                      ${isHighlighted ? 'bg-ctp-surface1' : 'hover:bg-ctp-surface1/50'}`}
                    onMouseEnter={() => setHighlightIndex(idx)}
                    onClick={() => handleSelect(preset)}
                  >
                    <span className="text-[13px] text-ctp-text">{preset.name}</span>
                    <span className="text-[11px] text-ctp-overlay0">{preset.desc}</span>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      )}
      {open && grouped.length === 0 && query.trim() && (
        <div className="absolute top-full left-0 right-0 mt-1 p-3 text-center
                        bg-ctp-surface0 border border-ctp-surface1 rounded-md z-[100]
                        shadow-xl shadow-black/30 animate-slide-down">
          <span className="text-ctp-overlay0 text-[13px]">No matching presets</span>
        </div>
      )}
    </div>
  )
}
