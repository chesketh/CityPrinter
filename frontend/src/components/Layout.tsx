import type { ReactNode } from 'react'

interface LayoutProps {
  left: ReactNode
  right: ReactNode
}

export default function Layout({ left, right }: LayoutProps) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-ctp-base text-ctp-text font-sans">
      <div className="flex flex-col w-[38%] min-w-[340px] bg-ctp-base border-r border-ctp-surface1/30">
        {left}
      </div>
      <div className="flex-1 bg-ctp-mantle flex items-center justify-center overflow-hidden">
        {right}
      </div>
    </div>
  )
}
