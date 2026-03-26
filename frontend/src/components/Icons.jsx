// Custom SVG icon set — used throughout the UI instead of emojis

function Icon({ size = 16, className = '', strokeWidth = 1.75, children, ...props }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`icon${className ? ' ' + className : ''}`}
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  )
}

export function ShieldIcon(p) {
  return <Icon {...p}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></Icon>
}

export function ShieldCheckIcon(p) {
  return (
    <Icon {...p}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      <polyline points="9 12 11 14 15 10"/>
    </Icon>
  )
}

export function LockIcon(p) {
  return (
    <Icon {...p}>
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
      <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
    </Icon>
  )
}

export function UnlockIcon(p) {
  return (
    <Icon {...p}>
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
      <path d="M7 11V7a5 5 0 0 1 9.9-1"/>
    </Icon>
  )
}

export function BanIcon(p) {
  return (
    <Icon {...p}>
      <circle cx="12" cy="12" r="10"/>
      <line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
    </Icon>
  )
}

export function BotIcon(p) {
  return (
    <Icon {...p}>
      <rect x="3" y="11" width="18" height="10" rx="2"/>
      <path d="M12 11V7"/>
      <circle cx="12" cy="5" r="2"/>
      <circle cx="8.5" cy="16" r="1" fill="currentColor" stroke="none"/>
      <circle cx="15.5" cy="16" r="1" fill="currentColor" stroke="none"/>
    </Icon>
  )
}

export function UserIcon(p) {
  return (
    <Icon {...p}>
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
      <circle cx="12" cy="7" r="4"/>
    </Icon>
  )
}

export function AlertTriangleIcon(p) {
  return (
    <Icon {...p}>
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <circle cx="12" cy="17" r="0.5" fill="currentColor" stroke="none"/>
    </Icon>
  )
}

export function CheckIcon(p) {
  return <Icon {...p}><polyline points="20 6 9 17 4 12"/></Icon>
}

export function SendIcon(p) {
  return (
    <Icon {...p}>
      <line x1="22" y1="2" x2="11" y2="13"/>
      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </Icon>
  )
}

export function ActivityIcon(p) {
  return <Icon {...p}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></Icon>
}

export function BarChartIcon(p) {
  return (
    <Icon {...p}>
      <line x1="12" y1="20" x2="12" y2="10"/>
      <line x1="18" y1="20" x2="18" y2="4"/>
      <line x1="6" y1="20" x2="6" y2="16"/>
    </Icon>
  )
}

export function MessageSquareIcon(p) {
  return (
    <Icon {...p}>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </Icon>
  )
}

export function XIcon(p) {
  return (
    <Icon {...p}>
      <line x1="18" y1="6" x2="6" y2="18"/>
      <line x1="6" y1="6" x2="18" y2="18"/>
    </Icon>
  )
}

export function ServerIcon(p) {
  return (
    <Icon {...p}>
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>
      <line x1="6" y1="6" x2="6.01" y2="6" strokeWidth="3"/>
      <line x1="6" y1="18" x2="6.01" y2="18" strokeWidth="3"/>
    </Icon>
  )
}

export function SlidersIcon(p) {
  return (
    <Icon {...p}>
      <line x1="4" y1="21" x2="4" y2="14"/>
      <line x1="4" y1="10" x2="4" y2="3"/>
      <line x1="12" y1="21" x2="12" y2="12"/>
      <line x1="12" y1="8" x2="12" y2="3"/>
      <line x1="20" y1="21" x2="20" y2="16"/>
      <line x1="20" y1="12" x2="20" y2="3"/>
      <line x1="1" y1="14" x2="7" y2="14"/>
      <line x1="9" y1="8" x2="15" y2="8"/>
      <line x1="17" y1="16" x2="23" y2="16"/>
    </Icon>
  )
}

export function GridIcon(p) {
  return (
    <Icon {...p}>
      <rect x="3" y="3" width="7" height="7"/>
      <rect x="14" y="3" width="7" height="7"/>
      <rect x="14" y="14" width="7" height="7"/>
      <rect x="3" y="14" width="7" height="7"/>
    </Icon>
  )
}

export function TrashIcon(p) {
  return (
    <Icon {...p}>
      <polyline points="3 6 5 6 21 6"/>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
    </Icon>
  )
}

export function TrendingUpIcon(p) {
  return (
    <Icon {...p}>
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
      <polyline points="17 6 23 6 23 12"/>
    </Icon>
  )
}

export function CpuIcon(p) {
  return (
    <Icon {...p}>
      <rect x="4" y="4" width="16" height="16" rx="2"/>
      <rect x="9" y="9" width="6" height="6"/>
      <line x1="9" y1="1" x2="9" y2="4"/>
      <line x1="15" y1="1" x2="15" y2="4"/>
      <line x1="9" y1="20" x2="9" y2="23"/>
      <line x1="15" y1="20" x2="15" y2="23"/>
      <line x1="20" y1="9" x2="23" y2="9"/>
      <line x1="20" y1="14" x2="23" y2="14"/>
      <line x1="1" y1="9" x2="4" y2="9"/>
      <line x1="1" y1="14" x2="4" y2="14"/>
    </Icon>
  )
}

export function TagIcon(p) {
  return (
    <Icon {...p}>
      <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>
      <line x1="7" y1="7" x2="7.01" y2="7" strokeWidth="3"/>
    </Icon>
  )
}

export function ChevronLeftIcon(p) {
  return <Icon {...p}><polyline points="15 18 9 12 15 6"/></Icon>
}

export function ChevronRightIcon(p) {
  return <Icon {...p}><polyline points="9 18 15 12 9 6"/></Icon>
}

export function SunIcon(p) {
  return (
    <Icon {...p}>
      <circle cx="12" cy="12" r="5"/>
      <line x1="12" y1="1" x2="12" y2="3"/>
      <line x1="12" y1="21" x2="12" y2="23"/>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
      <line x1="1" y1="12" x2="3" y2="12"/>
      <line x1="21" y1="12" x2="23" y2="12"/>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </Icon>
  )
}

export function MoonIcon(p) {
  return <Icon {...p}><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></Icon>
}

export function PlusIcon(p) {
  return (
    <Icon {...p}>
      <line x1="12" y1="5" x2="12" y2="19"/>
      <line x1="5" y1="12" x2="19" y2="12"/>
    </Icon>
  )
}

export function PencilIcon(p) {
  return (
    <Icon {...p}>
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
    </Icon>
  )
}

export function ZapIcon(p) {
  return (
    <Icon {...p}>
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </Icon>
  )
}

export function CopyIcon(p) {
  return (
    <Icon {...p}>
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
    </Icon>
  )
}

export function SyncScrollIcon(p) {
  return (
    <Icon {...p}>
      <path d="M8 3H5a2 2 0 0 0-2 2v3"/>
      <path d="M21 8V5a2 2 0 0 0-2-2h-3"/>
      <path d="M3 16v3a2 2 0 0 0 2 2h3"/>
      <path d="M16 21h3a2 2 0 0 0 2-2v-3"/>
      <line x1="12" y1="8" x2="12" y2="16"/>
      <line x1="8" y1="12" x2="16" y2="12"/>
    </Icon>
  )
}
