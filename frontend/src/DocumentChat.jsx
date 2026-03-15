import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import PropTypes from 'prop-types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle,
  ChevronDown,
  FileText,
  LoaderCircle,
  Sparkles,
} from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/ask'

const MotionDiv = motion.div
const MotionSpan = motion.span

// Matches text nodes that are only colons / whitespace (section-heading paragraph heuristic)
const SECTION_HEADING_PATTERN = /^[:\s]*$/

const MARKDOWN_COMPONENTS = {
  // ── Headings ──────────────────────────────────────────────────────────────
  h1: ({ children }) => (
    <h1 className="mb-4 mt-6 flex items-center gap-3 text-lg font-black text-zinc-50 first:mt-0">
      <span className="h-px flex-1 bg-gradient-to-r from-indigo-500/60 to-transparent" />
      <span className="bg-gradient-to-r from-indigo-300 to-fuchsia-300 bg-clip-text text-transparent">
        {children}
      </span>
      <span className="h-px flex-1 bg-gradient-to-l from-fuchsia-500/60 to-transparent" />
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="mb-2.5 mt-5 flex items-center gap-2.5 text-base font-bold text-zinc-100 first:mt-0">
      <span className="h-4 w-[3px] flex-none rounded-full bg-gradient-to-b from-indigo-400 to-fuchsia-400" />
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mb-2 mt-4 text-xs font-semibold uppercase tracking-widest text-indigo-300 first:mt-0">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="mb-1.5 mt-3 text-sm font-semibold text-zinc-200 first:mt-0">{children}</h4>
  ),
  // ── Paragraph (detects bold-only section headers) ─────────────────────────
  p: ({ children, node }) => {
    const childNodes = node?.children ?? []
    const isHeadingParagraph =
      childNodes.length > 0 &&
      childNodes.every(
        (c) =>
          (c.type === 'element' && c.tagName === 'strong') ||
          (c.type === 'text' && SECTION_HEADING_PATTERN.test(c.value ?? ''))
      )

    if (isHeadingParagraph) {
      return (
        <div className="mb-2 mt-5 flex items-center gap-2.5 border-l-[3px] border-indigo-500/70 pl-3 first:mt-0">
          <span className="text-sm font-bold text-zinc-100">{children}</span>
        </div>
      )
    }

    return (
      <p className="mb-2.5 text-sm leading-relaxed text-zinc-300 last:mb-0">{children}</p>
    )
  },
  // ── Lists ──────────────────────────────────────────────────────────────────
  ul: ({ children }) => <ul className="mb-3 space-y-1.5">{children}</ul>,
  ol: ({ children }) => <ol className="mb-3 space-y-1.5">{children}</ol>,
  li: ({ children, ordered, index }) => (
    <li className="flex items-start gap-3 text-zinc-200 [&>p]:mb-0 [&>p]:inline">
      {ordered ? (
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-indigo-500/25 text-xs font-semibold text-indigo-300">
          {(index ?? 0) + 1}
        </span>
      ) : (
        <span className="mt-[7px] h-[7px] w-[7px] shrink-0 rounded-full bg-gradient-to-b from-indigo-400 to-fuchsia-500" />
      )}
      <span className="min-w-0 flex-1 leading-relaxed">{children}</span>
    </li>
  ),
  // ── Inline ────────────────────────────────────────────────────────────────
  strong: ({ children }) => (
    <strong className="font-bold text-zinc-50">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-zinc-300/90">{children}</em>
  ),
  hr: () => <hr className="my-4 border-white/10" />,
  blockquote: ({ children }) => (
    <blockquote className="my-3 border-l-2 border-indigo-400/50 pl-4 italic text-zinc-300/90">
      {children}
    </blockquote>
  ),
  // ── Table ─────────────────────────────────────────────────────────────────
  table: (props) => (
    <div className="my-4 w-full overflow-x-auto rounded-xl border border-white/10">
      <table className="w-full border-collapse text-left" {...props} />
    </div>
  ),
  th: (props) => (
    <th
      className="border-b border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-zinc-300"
      {...props}
    />
  ),
  td: (props) => (
    <td
      className="border-b border-white/10 px-3 py-2 align-top text-sm text-zinc-300"
      {...props}
    />
  ),
  // ── Code ──────────────────────────────────────────────────────────────────
  code: ({ inline, className, children, ...props }) => {
    if (inline) {
      return (
        <code
          className="rounded bg-indigo-500/15 px-1.5 py-0.5 font-mono text-[0.9em] text-indigo-200"
          {...props}
        >
          {children}
        </code>
      )
    }

    return (
      <code className={className} {...props}>
        {children}
      </code>
    )
  },
  pre: (props) => (
    <pre
      className="my-4 overflow-x-auto rounded-xl border border-white/10 bg-black/50 p-4 font-mono text-sm text-zinc-200"
      {...props}
    />
  ),
}

function hashString(input) {
  let hash = 0
  for (const ch of input) {
    const codePoint = ch.codePointAt(0) ?? 0
    hash = (hash * 31 + codePoint) % 4294967296
  }
  return String(Math.trunc(hash))
}

/**
 * Strips PDF extraction artifacts from raw source chunk text.
 * Handles patterns like /enve[pecu, /github (before github.com), /]\18, etc.
 */
function cleanSourceText(raw) {
  return String(raw)
    // PDF artifact: /word where word contains bracket or backslash (font/encoding refs)
    .replace(/\/[A-Za-z]*[[\]\\][A-Za-z0-9[\]\\]*/g, '')
    // PDF artifact: /word immediately before a duplicate of that same word (hyperlink prefix)
    .replace(/\/([A-Za-z]{4,})(?=\1)/g, '')
    // Remove stray backslashes left behind
    .replace(/\\/g, '')
    // Collapse multiple spaces/tabs into one
    .replace(/[ \t]{2,}/g, ' ')
    // Max two consecutive newlines
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

async function askDocument(trimmedQuestion, signal) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question: trimmedQuestion }),
    signal,
  })

  if (!response.ok) {
    const maybeText = await response.text().catch(() => '')
    const message = maybeText
      ? `Request failed (${response.status}): ${maybeText}`
      : `Request failed (${response.status})`
    throw new Error(message)
  }

  return response.json()
}

function toUserFacingError(fetchError) {
  const isAbort = fetchError instanceof DOMException && fetchError.name === 'AbortError'
  if (isAbort) {
    return 'Request timed out. Is the FastAPI server responding at http://127.0.0.1:8000?'
  }

  return 'Could not reach the backend. Is FastAPI running at http://127.0.0.1:8000?'
}

async function submitQuestion({
  question,
  setIsLoading,
  setError,
  setResult,
  setActiveSource,
  setSubmittedQuestion,
}) {
  const trimmed = question.trim()
  if (!trimmed) {
    setError('Type a question first.')
    return
  }

  setIsLoading(true)
  setError(null)
  setResult(null)
  setActiveSource(null)
  setSubmittedQuestion(trimmed)

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 25000)

  try {
    const data = await askDocument(trimmed, controller.signal)
    setResult(data)
  } catch (fetchError) {
    console.error('Error communicating with backend:', fetchError)
    setError(toUserFacingError(fetchError))
  } finally {
    clearTimeout(timeoutId)
    setIsLoading(false)
  }
}

function SkeletonLines() {
  const widths = ['w-10/12', 'w-11/12', 'w-9/12', 'w-11/12', 'w-8/12']

  return (
    <MotionDiv
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="space-y-3"
      aria-label="Loading"
    >
      <div className="flex items-center gap-2 text-xs text-zinc-300/80">
        <LoaderCircle className="h-4 w-4 animate-spin" />
        <span>Thinking…</span>
      </div>

      <MotionDiv
        animate={{ opacity: [0.45, 0.95, 0.45] }}
        transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
        className="space-y-2"
      >
        {widths.map((w, idx) => (
          <div key={`${w}-${idx}`} className={`h-3 ${w} rounded-full bg-white/10`} />
        ))}
        <div className="h-3 w-7/12 rounded-full bg-white/10" />
      </MotionDiv>
    </MotionDiv>
  )
}

function MarkdownAnswer({ markdown }) {
  return (
    <div className="w-full">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
        {markdown}
      </ReactMarkdown>
    </div>
  )
}

function ErrorBanner({ error }) {
  if (!error) return null

  return (
    <MotionDiv
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      className="mt-4 flex items-start gap-3 rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-100"
      role="alert"
    >
      <AlertTriangle className="mt-0.5 h-5 w-5 flex-none text-red-200" />
      <div className="leading-relaxed">
        <div className="font-semibold">Something went wrong</div>
        <div className="text-red-100/80">{error}</div>
      </div>
    </MotionDiv>
  )
}

function SourcesSection({ sources, activeSource, setActiveSource }) {
  if (!sources || sources.length === 0) {
    return (
      <div className="mt-6 border-t border-white/10 pt-4">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs font-semibold tracking-wider text-zinc-300/70">SOURCES</div>
          <div className="text-xs text-zinc-400/70">Click to expand</div>
        </div>
        <div className="mt-3 text-xs text-zinc-400/70">No sources returned.</div>
      </div>
    )
  }

  const activeSourceItem =
    activeSource != null && sources[activeSource] ? sources[activeSource] : null

  return (
    <div className="mt-6 border-t border-white/10 pt-4">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs font-semibold tracking-wider text-zinc-300/70">SOURCES</div>
        <div className="text-xs text-zinc-400/70">Click to expand</div>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {sources.map((sourceItem, idx) => {
          const isActive = activeSource === idx
          return (
            <button
              key={sourceItem.key}
              type="button"
              onClick={() => setActiveSource(isActive ? null : idx)}
              className={
                isActive
                  ? 'inline-flex items-center gap-2 rounded-full border border-indigo-400/40 bg-indigo-500/15 px-3 py-1.5 text-xs font-semibold text-zinc-100'
                  : 'inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-semibold text-zinc-200/90 hover:bg-white/10'
              }
            >
              <FileText className="h-4 w-4 text-indigo-200" />
              Source {idx + 1}
              <MotionSpan
                animate={{ rotate: isActive ? 180 : 0 }}
                transition={{ duration: 0.18 }}
                className="inline-flex"
              >
                <ChevronDown className="h-4 w-4 text-zinc-200/70" />
              </MotionSpan>
            </button>
          )
        })}
      </div>

      <AnimatePresence>
        {activeSourceItem ? (
          <MotionDiv
            key={`active-source-${activeSourceItem.key}`}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            className="mt-3 overflow-hidden"
          >
            <div className="rounded-2xl border border-white/10 bg-black/30 p-4">
              <div className="mb-2 flex items-center justify-between gap-3">
                <div className="text-xs font-semibold text-zinc-200">
                  Source {activeSource + 1}
                </div>
                <button
                  type="button"
                  onClick={() => setActiveSource(null)}
                  className="text-xs text-zinc-400/80 hover:text-zinc-200"
                >
                  Close
                </button>
              </div>
              <div className="max-h-64 overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-zinc-300/90">
                {cleanSourceText(activeSourceItem.text)}
              </div>
            </div>
          </MotionDiv>
        ) : null}
      </AnimatePresence>
    </div>
  )
}

function Transcript({ submittedQuestion, isLoading, answer, error, sources, activeSource, setActiveSource }) {
  const show = submittedQuestion || isLoading || answer || error
  if (!show) return null

  let answerNode = <div className="text-sm text-zinc-300/80">No answer yet.</div>
  if (isLoading) {
    answerNode = <SkeletonLines />
  } else if (answer) {
    answerNode = <MarkdownAnswer markdown={answer} />
  }

  return (
    <div className="mt-7 space-y-4">
      <div className="rounded-3xl border border-white/10 bg-black/20 p-4 backdrop-blur sm:p-5">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-9 w-9 flex-none items-center justify-center rounded-2xl bg-white/5 text-xs font-bold text-zinc-200">
            You
          </div>
          <div className="min-w-0 flex-1 text-sm leading-relaxed text-zinc-100">
            {submittedQuestion}
          </div>
        </div>
      </div>

      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 shadow-[0_10px_50px_rgba(0,0,0,0.35)] backdrop-blur-xl sm:p-5">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-9 w-9 flex-none items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500/30 to-fuchsia-500/30 text-xs font-bold text-zinc-100">
            AI
          </div>

          <div className="min-w-0 flex-1">{answerNode}</div>
        </div>

        {!isLoading && answer ? (
          <SourcesSection
            sources={sources}
            activeSource={activeSource}
            setActiveSource={setActiveSource}
          />
        ) : null}
      </div>
    </div>
  )
}

function DocumentChat() {
  const [question, setQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [activeSource, setActiveSource] = useState(null)
  const [error, setError] = useState(null)

  const sources = useMemo(() => {
    const raw = result?.sources_used
    const arr = Array.isArray(raw) ? raw : []

    const seen = new Map()
    return arr.map((text) => {
      const base = `${hashString(String(text))}-${String(text).length}`
      const count = seen.get(base) ?? 0
      seen.set(base, count + 1)
      return {
        key: `${base}-${count}`,
        text,
      }
    })
  }, [result?.sources_used])

  const canSubmit = !isLoading && question.trim().length > 0

  const handleAsk = (event) => {
    event.preventDefault()
    if (isLoading) return

    void submitQuestion({
      question,
      setIsLoading,
      setError,
      setResult,
      setActiveSource,
      setSubmittedQuestion,
    })
  }

  const answer = result?.answer ?? ''

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-zinc-950 via-zinc-950 to-indigo-950">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-24 -top-24 h-72 w-72 rounded-full bg-fuchsia-500/10 blur-3xl" />
        <div className="absolute -right-24 top-24 h-72 w-72 rounded-full bg-indigo-500/10 blur-3xl" />
      </div>

      <div className="relative mx-auto w-full max-w-5xl px-4 py-12">
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-[0_25px_80px_rgba(0,0,0,0.55)] backdrop-blur-xl sm:p-8">
          <header className="flex flex-col gap-3">
            <div className="flex items-center gap-2 text-xs font-semibold tracking-wider text-zinc-300/80">
              <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-3 py-1">
                <Sparkles className="h-4 w-4 text-fuchsia-300" />
                Premium RAG Chat
              </span>
            </div>

            <h1 className="text-3xl font-black tracking-tight text-zinc-50 sm:text-4xl">
              Chat with my Resume
            </h1>
            <p className="max-w-2xl text-sm leading-relaxed text-zinc-300/80">
              Ask any question from my resume. The assistant answers using only retrieved chunks and shows citations so you can verify.
            </p>
          </header>

          <form onSubmit={handleAsk} className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="relative flex-1">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask anything about your document…"
                aria-label="Question"
                spellCheck={false}
                autoCorrect="off"
                autoCapitalize="off"
                disabled={isLoading}
                className="w-full rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-zinc-100 outline-none ring-0 placeholder:text-zinc-500 focus:border-indigo-400/50 focus:bg-black/40 focus:shadow-[0_0_0_4px_rgba(99,102,241,0.15)] disabled:opacity-60"
              />
            </div>

            <button
              type="submit"
              disabled={!canSubmit}
              className="group inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-5 py-3 text-sm font-semibold text-white shadow-[0_10px_30px_rgba(99,102,241,0.35)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoading ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  Asking…
                </>
              ) : (
                <>Ask</>
              )}
            </button>
          </form>

          <AnimatePresence>
            <ErrorBanner error={error} />
          </AnimatePresence>

          <Transcript
            submittedQuestion={submittedQuestion}
            isLoading={isLoading}
            answer={answer}
            error={error}
            sources={sources}
            activeSource={activeSource}
            setActiveSource={setActiveSource}
          />

          <footer className="mt-8 text-center text-xs text-zinc-500">
            Backend: <span className="text-zinc-300/70">{API_URL}</span>
          </footer>
        </div>
      </div>
    </div>
  )
}

export default DocumentChat

MarkdownAnswer.propTypes = {
  markdown: PropTypes.string.isRequired,
}

const SourceItemPropType = PropTypes.shape({
  key: PropTypes.string.isRequired,
  text: PropTypes.any.isRequired,
})

ErrorBanner.propTypes = {
  error: PropTypes.string,
}

ErrorBanner.defaultProps = {
  error: null,
}

SourcesSection.propTypes = {
  sources: PropTypes.arrayOf(SourceItemPropType).isRequired,
  activeSource: PropTypes.number,
  setActiveSource: PropTypes.func.isRequired,
}

SourcesSection.defaultProps = {
  activeSource: null,
}

Transcript.propTypes = {
  submittedQuestion: PropTypes.string,
  isLoading: PropTypes.bool.isRequired,
  answer: PropTypes.string,
  error: PropTypes.string,
  sources: PropTypes.arrayOf(SourceItemPropType).isRequired,
  activeSource: PropTypes.number,
  setActiveSource: PropTypes.func.isRequired,
}

Transcript.defaultProps = {
  submittedQuestion: '',
  answer: '',
  error: null,
  activeSource: null,
}
