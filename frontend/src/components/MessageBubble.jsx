import { useState } from 'react'
import { BotIcon, UserIcon, BanIcon, AlertTriangleIcon, CheckIcon, CopyIcon, ChevronRightIcon } from './Icons'

const COLLAPSE_THRESHOLD = 400 // chars

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      // clipboard not available
    }
  }

  return (
    <button
      className={`copy-btn ${copied ? 'copy-btn-copied' : ''}`}
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Copy message'}
      aria-label="Copy message"
    >
      {copied ? <CheckIcon size={11} /> : <CopyIcon size={11} />}
    </button>
  )
}

function ProbBar({ probability }) {
  if (probability == null) return null
  const pct = Math.round(probability * 100)
  const colorClass = probability > 0.8 ? 'prob-bar-high' : probability > 0.5 ? 'prob-bar-medium' : 'prob-bar-low'
  return (
    <div className="prob-bar-wrap" title={`Injection probability: ${pct}%`}>
      <div className={`prob-bar-fill ${colorClass}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export default function MessageBubble({ message }) {
  const { role, content, type, probability, detector, llm, detect_ms, total_ms } = message
  const [expanded, setExpanded] = useState(false)

  if (role === 'user') {
    return (
      <div className="bubble-row bubble-row-user">
        <div className="bubble bubble-user">
          <span className="bubble-role">
            <UserIcon size={11} />
            You
          </span>
          <p className="bubble-text">{content}</p>
        </div>
      </div>
    )
  }

  const isBlocked = type === 'blocked'

  return (
    <div className="bubble-row bubble-row-assistant">
      <div className={`bubble ${isBlocked ? 'bubble-blocked' : 'bubble-assistant'}`}>
        {/* Header row */}
        <div className="bubble-meta">
          <span className="bubble-role">
            {isBlocked
              ? <><BanIcon size={11} /> Blocked</>
              : <><BotIcon size={11} /> Assistant</>
            }
          </span>

          {/* Injection probability badge */}
          {probability !== undefined && (
            <span
              className={`prob-badge ${
                probability > 0.8 ? 'prob-high' :
                probability > 0.5 ? 'prob-medium' :
                'prob-low'
              }`}
              title={`Injection probability: ${(probability * 100).toFixed(1)}%`}
            >
              {isBlocked
                ? <AlertTriangleIcon size={10} />
                : <CheckIcon size={10} />
              }
              {(probability * 100).toFixed(1)}%
            </span>
          )}

          {/* Detector tag */}
          {detector && detector !== 'none' && (
            <span className="tag tag-detector">{detector}</span>
          )}

          {/* LLM tag (only for forwarded responses) */}
          {llm && !isBlocked && (
            <span className="tag tag-llm">{llm}</span>
          )}

          {/* Copy button — pushed to right */}
          {content && <CopyButton text={content} />}
        </div>

        {/* Probability bar */}
        <ProbBar probability={probability} />

        {/* Message body — collapsible when long */}
        {content && content.length > COLLAPSE_THRESHOLD && !expanded ? (
          <>
            <p className="bubble-text bubble-text-clamped">{content}</p>
            <button className="bubble-expand-btn" onClick={() => setExpanded(true)}>
              <ChevronRightIcon size={11} />
              Show more
            </button>
          </>
        ) : (
          <>
            <p className="bubble-text">{content}</p>
            {content && content.length > COLLAPSE_THRESHOLD && (
              <button className="bubble-expand-btn" onClick={() => setExpanded(false)}>
                <ChevronRightIcon size={11} style={{ transform: 'rotate(90deg)' }} />
                Show less
              </button>
            )}
          </>
        )}

        {/* Footer timings */}
        {(detect_ms !== undefined || total_ms !== undefined) && (
          <div className="bubble-timings">
            {detect_ms > 0 && <span>Detect: {detect_ms.toFixed(0)}ms</span>}
            {total_ms  > 0 && <span>Total: {total_ms.toFixed(0)}ms</span>}
          </div>
        )}
      </div>
    </div>
  )
}
