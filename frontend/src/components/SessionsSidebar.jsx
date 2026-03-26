import { useState } from 'react'
import { PlusIcon, TrashIcon, PencilIcon, MessageSquareIcon, ChevronLeftIcon, ChevronRightIcon } from './Icons'

function formatAge(isoString) {
  if (!isoString) return ''
  const diff = Date.now() - new Date(isoString).getTime()
  const mins  = Math.floor(diff / 60_000)
  const hours = Math.floor(diff / 3_600_000)
  const days  = Math.floor(diff / 86_400_000)
  if (mins  < 1)  return 'just now'
  if (mins  < 60) return `${mins}m ago`
  if (hours < 24) return `${hours}h ago`
  if (days  < 7)  return `${days}d ago`
  return new Date(isoString).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function SearchIcon(p) {
  return (
    <svg width={p.size ?? 13} height={p.size ?? 13} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
  )
}

export default function SessionsSidebar({
  sessions,
  activeSessionId,
  onSelect,
  onCreate,
  onDelete,
  onRename,
  collapsed,
  onToggleCollapse,
}) {
  const [editingId, setEditingId] = useState(null)
  const [editValue, setEditValue] = useState('')
  const [query,     setQuery]     = useState('')

  function startEdit(e, session) {
    e.stopPropagation()
    setEditingId(session.id)
    setEditValue(session.title)
  }

  function commitEdit(session) {
    const trimmed = editValue.trim()
    if (trimmed && trimmed !== session.title) onRename(session.id, trimmed)
    setEditingId(null)
  }

  function handleEditKeyDown(e, session) {
    if (e.key === 'Enter')  { e.preventDefault(); commitEdit(session) }
    if (e.key === 'Escape') { setEditingId(null) }
  }

  const filtered = query.trim()
    ? sessions.filter(s => s.title.toLowerCase().includes(query.toLowerCase()))
    : sessions

  return (
    <aside className={`sessions-sidebar${collapsed ? ' collapsed' : ''}`}>
      <div className="sessions-sidebar-header">
        {!collapsed && (
          <>
            <span className="sessions-sidebar-title">Conversations</span>
            <button className="sessions-new-btn" onClick={onCreate} title="New conversation">
              <PlusIcon size={13} />
            </button>
          </>
        )}
        <button
          className="sessions-collapse-btn"
          onClick={onToggleCollapse}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRightIcon size={14} /> : <ChevronLeftIcon size={14} />}
        </button>
      </div>

      {collapsed ? (
        <div className="sessions-collapsed-actions">
          <button className="sessions-new-btn" onClick={onCreate} title="New conversation">
            <PlusIcon size={14} />
          </button>
        </div>
      ) : (
        <>
          {/* Search box — only shown when there are sessions */}
          {sessions.length > 0 && (
            <div className="sessions-search-wrap">
              <SearchIcon size={12} />
              <input
                className="sessions-search"
                placeholder="Search…"
                value={query}
                onChange={e => setQuery(e.target.value)}
              />
              {query && (
                <button className="sessions-search-clear" onClick={() => setQuery('')} aria-label="Clear search">
                  ×
                </button>
              )}
            </div>
          )}

          <div className="sessions-list">
            {filtered.length === 0 ? (
              <div className="sessions-empty-state">
                {sessions.length === 0
                  ? <><MessageSquareIcon size={24} /><span>No saved chats yet</span></>
                  : <span className="sessions-no-match">No matches</span>
                }
              </div>
            ) : (
              filtered.map(s => (
                <div
                  key={s.id}
                  className={`session-item ${s.id === activeSessionId ? 'session-item-active' : ''}`}
                  onClick={() => onSelect(s.id)}
                >
                  <div className="session-item-body">
                    {editingId === s.id ? (
                      <input
                        className="session-rename-input"
                        value={editValue}
                        autoFocus
                        onChange={e => setEditValue(e.target.value)}
                        onBlur={() => commitEdit(s)}
                        onKeyDown={e => handleEditKeyDown(e, s)}
                        onClick={e => e.stopPropagation()}
                      />
                    ) : (
                      <span className="session-title">{s.title}</span>
                    )}
                    <div className="session-meta">
                      <span>{s.message_count ?? 0} msg{s.message_count !== 1 ? 's' : ''}</span>
                      <span className="session-age">{formatAge(s.updated_at)}</span>
                    </div>
                  </div>

                  <div className="session-item-actions">
                    <button
                      className="session-action-btn"
                      title="Rename"
                      onClick={e => startEdit(e, s)}
                    >
                      <PencilIcon size={11} />
                    </button>
                    <button
                      className="session-action-btn session-action-delete"
                      title="Delete conversation"
                      onClick={e => { e.stopPropagation(); onDelete(s.id) }}
                    >
                      <TrashIcon size={11} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </>
      )}
    </aside>
  )
}
