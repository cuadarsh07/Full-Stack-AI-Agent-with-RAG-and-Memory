import { useEffect, useMemo, useRef, useState } from 'react'
import PropTypes from 'prop-types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  AlertTriangle,
  Bot,
  FileText,
  Globe,
  Loader2,
  MessageSquareText,
  Plus,
  Send,
  ShieldAlert,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  User,
} from 'lucide-react'

const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'https://ai-agent-backend-l44r.onrender.com'
const SESSION_STORAGE_KEY = 'adarsh-ai-session-id'
const MESSAGE_STORAGE_KEY = 'adarsh-ai-visible-messages'
const FALLBACK_ASSISTANT_REPLY = "I'm here and ready to help."

const STARTER_PROMPTS = [
  "Give me a short overview of Adarsh Kumar's background.",
  "What are Adarsh's strongest technical skills?",
  'What is happening in AI today?',
]

const FUNCTION_BLOCK_PATTERN = /<function[\s\S]*?<\/function>/gi
const TOOL_BLOCK_PATTERN = /<tool_call[\s\S]*?<\/tool_call>/gi
const XML_DECLARATION_PATTERN = /<\?xml[\s\S]*?\?>/gi
const XML_TAG_PATTERN = /<\/?[a-z_][\w:.-]*(?:\s+[^<>]*?)?>/gi
const ENCODED_FUNCTION_TAG_PATTERN = /&lt;\/?function.*?&gt;/gi

const MARKDOWN_COMPONENTS = {
  p: ({ children }) => <p className="mb-3 last:mb-0 leading-7 text-zinc-100/95">{children}</p>,
  ul: ({ children }) => <ul className="mb-3 list-disc space-y-2 pl-5 text-zinc-100/95">{children}</ul>,
  ol: ({ children }) => <ol className="mb-3 list-decimal space-y-2 pl-5 text-zinc-100/95">{children}</ol>,
  li: ({ children }) => <li>{children}</li>,
  code: ({ inline, children }) =>
    inline ? (
      <code className="rounded bg-black/30 px-1.5 py-0.5 text-[0.95em] text-amber-100">{children}</code>
    ) : (
      <code className="block overflow-x-auto rounded-2xl border border-white/10 bg-black/40 p-4 text-sm text-zinc-100">{children}</code>
    ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-cyan-300 underline decoration-cyan-500/50 underline-offset-4"
    >
      {children}
    </a>
  ),
  strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
}

function makeClientId(prefix = 'msg') {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function makeSessionId() {
  return makeClientId('session')
}

function getToolLabel(toolName, fallbackLabel) {
  if (fallbackLabel) {
    return fallbackLabel
  }

  if (toolName === 'search_live_web') {
    return 'Searched the web'
  }

  if (toolName === 'search_resume_database') {
    return 'Read resume'
  }

  return 'Used a source'
}

function sanitizeAssistantContent(content) {
  const sanitized = String(content ?? '')
    .replaceAll(FUNCTION_BLOCK_PATTERN, ' ')
    .replaceAll(TOOL_BLOCK_PATTERN, ' ')
    .replaceAll(XML_DECLARATION_PATTERN, ' ')
    .replaceAll(ENCODED_FUNCTION_TAG_PATTERN, ' ')
    .replaceAll(XML_TAG_PATTERN, ' ')
    .replaceAll(/[ \t]{2,}/g, ' ')
    .replaceAll(/\n{3,}/g, '\n\n')
    .trim()

  return sanitized || FALLBACK_ASSISTANT_REPLY
}

function normalizeToolsUsed(value) {
  if (!Array.isArray(value)) {
    return []
  }

  const seen = new Set()

  return value
    .filter((tool) => tool && typeof tool === 'object')
    .map((tool) => ({
      name: typeof tool.name === 'string' ? tool.name : 'unknown_tool',
      label: getToolLabel(
        typeof tool.name === 'string' ? tool.name : 'unknown_tool',
        typeof tool.label === 'string' ? tool.label : '',
      ),
      query: typeof tool.query === 'string' ? tool.query : '',
      status: typeof tool.status === 'string' ? tool.status : 'success',
    }))
    .filter((tool) => {
      const key = `${tool.name}:${tool.label}`
      if (seen.has(key)) {
        return false
      }

      seen.add(key)
      return true
    })
}

function normalizeStoredMessages(value) {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .filter((message) => message && typeof message.content === 'string' && typeof message.role === 'string')
    .map((message) => ({
      id: typeof message.id === 'string' ? message.id : makeClientId(),
      role: message.role,
      content:
        message.role === 'assistant' ? sanitizeAssistantContent(message.content) : message.content,
      toolsUsed: normalizeToolsUsed(message.toolsUsed),
      escalated: Boolean(message.escalated),
      warnings: Array.isArray(message.warnings) ? message.warnings.filter(Boolean) : [],
      feedback: typeof message.feedback === 'boolean' ? message.feedback : null,
    }))
}

function loadStoredMessages() {
  try {
    const raw = localStorage.getItem(MESSAGE_STORAGE_KEY)
    return raw ? normalizeStoredMessages(JSON.parse(raw)) : []
  } catch (error) {
    console.error('Unable to restore chat messages:', error)
    return []
  }
}

function toolAppearance(tool) {
  if (tool.name === 'search_live_web') {
    return {
      icon: Globe,
      className: 'border-sky-400/30 bg-sky-400/10 text-sky-100',
    }
  }

  if (tool.name === 'search_resume_database') {
    return {
      icon: FileText,
      className: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-100',
    }
  }

  return {
    icon: Sparkles,
    className: 'border-amber-400/30 bg-amber-400/10 text-amber-100',
  }
}

function ToolBadge({ tool }) {
  const appearance = toolAppearance(tool)
  const Icon = appearance.icon

  return (
    <span
      title={tool.query || tool.label}
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${appearance.className}`}
    >
      <Icon className="h-3.5 w-3.5" />
      <span>{tool.label}</span>
      {tool.status === 'error' && <span className="text-[10px] uppercase tracking-[0.2em]">Error</span>}
    </span>
  )
}

function AssistantMarkdown({ content }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
      {content}
    </ReactMarkdown>
  )
}

ToolBadge.propTypes = {
  tool: PropTypes.shape({
    name: PropTypes.string.isRequired,
    label: PropTypes.string.isRequired,
    query: PropTypes.string,
    status: PropTypes.string,
  }).isRequired,
}

AssistantMarkdown.propTypes = {
  content: PropTypes.string.isRequired,
}

export default function Chat() {
  const [messages, setMessages] = useState(() => loadStoredMessages())
  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem(SESSION_STORAGE_KEY) ?? makeSessionId(),
  )
  const [input, setInput] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [pendingFeedback, setPendingFeedback] = useState({})
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  const totalAssistantMessages = useMemo(
    () => messages.filter((message) => message.role === 'assistant').length,
    [messages],
  )
  const hasMessages = messages.length > 0

  useEffect(() => {
    localStorage.setItem(MESSAGE_STORAGE_KEY, JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    if (sessionId) {
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId)
    }
  }, [sessionId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleNewChat = () => {
    const nextSessionId = makeSessionId()
    setMessages([])
    setInput('')
    setError('')
    setPendingFeedback({})
    setSessionId(nextSessionId)
    localStorage.removeItem(MESSAGE_STORAGE_KEY)
    requestAnimationFrame(() => textareaRef.current?.focus())
  }

  const handlePromptSelection = (prompt) => {
    setInput(prompt)
    requestAnimationFrame(() => textareaRef.current?.focus())
  }

  const handleSendMessage = async (event) => {
    event.preventDefault()
    const trimmedInput = input.trim()
    if (!trimmedInput || isLoading) {
      return
    }

    const userMessage = {
      id: makeClientId(),
      role: 'user',
      content: trimmedInput,
      toolsUsed: [],
      escalated: false,
      warnings: [],
      feedback: null,
    }

    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setInput('')
    setError('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/master-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          messages: updatedMessages.map((message) => ({
            role: message.role,
            content: message.content,
          })),
        }),
      })

      if (!response.ok) {
        const details = await response.text().catch(() => '')
        throw new Error(details || `Request failed with status ${response.status}`)
      }

      const data = await response.json()
      setSessionId(typeof data.session_id === 'string' ? data.session_id : sessionId)
      setMessages((previousMessages) => [
        ...previousMessages,
        {
          id: typeof data.message_id === 'string' ? data.message_id : makeClientId('assistant'),
          role: 'assistant',
          content: sanitizeAssistantContent(data.answer),
          toolsUsed: normalizeToolsUsed(data.tools_used),
          escalated: Boolean(data.escalated),
          warnings: Array.isArray(data.warnings) ? data.warnings.filter(Boolean) : [],
          feedback: null,
        },
      ])
    } catch (requestError) {
      console.error('Master chat request failed:', requestError)
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Unable to reach the backend. Check VITE_API_URL or the FastAPI server.',
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleFeedback = async (messageId, thumbsUp, index) => {
    const targetMessage = messages[index]
    if (targetMessage?.feedback !== null || pendingFeedback?.[messageId]) {
      return
    }

    const previousUserMessage = messages
      .slice(0, index)
      .reverse()
      .find((message) => message.role === 'user')

    setPendingFeedback((current) => ({ ...current, [messageId]: true }))

    try {
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message_id: messageId,
          thumbs_up: thumbsUp,
          session_id: sessionId,
          user_message: previousUserMessage?.content ?? null,
          assistant_message: targetMessage.content,
          tools_used: targetMessage.toolsUsed,
        }),
      })

      if (!response.ok) {
        const details = await response.text().catch(() => '')
        throw new Error(details || `Feedback request failed with status ${response.status}`)
      }

      setMessages((currentMessages) =>
        currentMessages.map((message) =>
          message.id === messageId ? { ...message, feedback: thumbsUp } : message,
        ),
      )
    } catch (feedbackError) {
      console.error('Feedback request failed:', feedbackError)
      setError(
        feedbackError instanceof Error
          ? feedbackError.message
          : 'Unable to submit feedback right now.',
      )
    } finally {
      setPendingFeedback((current) => {
        const next = { ...current }
        delete next[messageId]
        return next
      })
    }
  }

  return (
    <div className="flex h-full min-h-0 w-full flex-col overflow-hidden rounded-[32px] border border-white/10 bg-[linear-gradient(180deg,_rgba(11,27,34,0.94)_0%,_rgba(7,18,24,0.98)_100%)] shadow-[0_32px_100px_rgba(0,0,0,0.38)]">
      <div className={`flex-shrink-0 border-b border-white/10 bg-white/[0.03] px-5 sm:px-6 ${hasMessages ? 'py-3' : 'py-4 sm:py-5'}`}>
        <div className={`flex flex-col gap-3 ${hasMessages ? 'xl:flex-row xl:items-center xl:justify-end' : 'xl:flex-row xl:items-end xl:justify-between'}`}>
          {!hasMessages && (
            <div className="max-w-2xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.06] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.26em] text-stone-300">
                <MessageSquareText className="h-3.5 w-3.5" />
                Conversation
              </div>
              <h2 className="mt-2 text-xl font-black tracking-tight text-white sm:text-[1.7rem]">
                Ask naturally. Get a clear answer.
              </h2>
              <p className="mt-1.5 max-w-xl text-sm leading-6 text-zinc-300 sm:text-[14px]">
                Ask about Adarsh's background, explore his resume, or check the latest news. The assistant keeps the routing behind the scenes and the experience simple.
              </p>
            </div>
          )}

          <div className={`flex flex-wrap items-center gap-3 text-xs text-zinc-300 ${hasMessages ? 'justify-between' : 'xl:justify-end'}`}>
            <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1.5">
              {hasMessages ? 'Conversation in progress' : 'Fresh chat ready'}
            </span>
            <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1.5">
              {totalAssistantMessages} replies saved
            </span>
            <button
              type="button"
              onClick={handleNewChat}
              className="inline-flex items-center justify-center gap-2 rounded-2xl bg-amber-300 px-6 py-3 text-sm font-bold text-slate-950 shadow-[0_16px_36px_rgba(252,211,77,0.26)] transition hover:bg-amber-200 focus:outline-none focus:ring-2 focus:ring-amber-200/60 focus:ring-offset-2 focus:ring-offset-slate-950 sm:text-[15px]"
            >
              <Plus className="h-4 w-4" />
              New Chat
            </button>
          </div>
        </div>

        {error && (
          <div className={`flex items-start gap-3 rounded-2xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-sm text-amber-100 ${hasMessages ? 'mt-3' : 'mt-4'}`}>
            <AlertTriangle className="mt-0.5 h-4 w-4 flex-none" />
            <span>{error}</span>
          </div>
        )}
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className={`flex-1 overflow-y-auto px-4 sm:px-6 ${hasMessages ? 'py-4' : 'py-5'}`}>
          {hasMessages ? (
            <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
              {messages.map((message, index) => {
                const isAssistant = message.role === 'assistant'
                const isFeedbackPending = Boolean(pendingFeedback[message.id])

                return (
                  <div
                    key={message.id}
                    className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {isAssistant && (
                      <div className="mt-1 flex h-10 w-10 flex-none items-center justify-center rounded-2xl bg-amber-300/12 text-amber-100 shadow-[0_0_0_1px_rgba(252,211,77,0.14)]">
                        <Bot className="h-5 w-5" />
                      </div>
                    )}

                    <div
                      className={`max-w-[88%] rounded-[28px] border px-5 py-4 shadow-[0_20px_60px_rgba(0,0,0,0.18)] sm:max-w-[78%] ${
                        message.role === 'user'
                          ? 'border-sky-400/20 bg-sky-400/12 text-white'
                          : 'border-white/10 bg-white/[0.04] text-zinc-100'
                      }`}
                    >
                      {message.role === 'user' ? (
                        <p className="whitespace-pre-wrap leading-7 text-white">{message.content}</p>
                      ) : (
                        <AssistantMarkdown content={message.content} />
                      )}

                      {isAssistant && message.toolsUsed.length > 0 && (
                        <div className="mt-4 flex flex-wrap gap-2 border-t border-white/10 pt-4">
                          {message.toolsUsed.map((tool) => (
                            <ToolBadge key={`${message.id}-${tool.name}-${tool.label}`} tool={tool} />
                          ))}
                        </div>
                      )}

                      {isAssistant && message.escalated && (
                        <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-amber-400/30 bg-amber-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-amber-100">
                          <ShieldAlert className="h-3.5 w-3.5" />
                          Needs more verified info
                        </div>
                      )}

                      {isAssistant && message.warnings.length > 0 && (
                        <div className="mt-4 rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-xs leading-6 text-zinc-400">
                          {message.warnings.join(' ')}
                        </div>
                      )}

                      {isAssistant && message.id && (
                        <div className="mt-4 flex items-center gap-2 border-t border-white/10 pt-4">
                          <button
                            type="button"
                            onClick={() => handleFeedback(message.id, true, index)}
                            disabled={isFeedbackPending || message.feedback !== null}
                            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold transition ${
                              message.feedback === true
                                ? 'border-emerald-400/40 bg-emerald-400/15 text-emerald-100'
                                : 'border-white/10 bg-white/[0.03] text-zinc-300 hover:border-emerald-400/30 hover:text-emerald-100'
                            } disabled:cursor-not-allowed disabled:opacity-70`}
                          >
                            <ThumbsUp className="h-3.5 w-3.5" />
                            Helpful
                          </button>
                          <button
                            type="button"
                            onClick={() => handleFeedback(message.id, false, index)}
                            disabled={isFeedbackPending || message.feedback !== null}
                            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold transition ${
                              message.feedback === false
                                ? 'border-rose-400/40 bg-rose-400/15 text-rose-100'
                                : 'border-white/10 bg-white/[0.03] text-zinc-300 hover:border-rose-400/30 hover:text-rose-100'
                            } disabled:cursor-not-allowed disabled:opacity-70`}
                          >
                            <ThumbsDown className="h-3.5 w-3.5" />
                            Needs work
                          </button>
                          {isFeedbackPending && <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />}
                        </div>
                      )}
                    </div>

                    {message.role === 'user' && (
                      <div className="mt-1 flex h-10 w-10 flex-none items-center justify-center rounded-2xl bg-sky-400/15 text-sky-100 shadow-[0_0_0_1px_rgba(56,189,248,0.18)]">
                        <User className="h-5 w-5" />
                      </div>
                    )}
                  </div>
                )
              })}

              {isLoading && (
                <div className="flex justify-start gap-4">
                  <div className="mt-1 flex h-10 w-10 flex-none items-center justify-center rounded-2xl bg-amber-300/12 text-amber-100 shadow-[0_0_0_1px_rgba(252,211,77,0.14)]">
                    <Loader2 className="h-5 w-5 animate-spin" />
                  </div>
                  <div className="max-w-[78%] rounded-[28px] border border-white/10 bg-white/[0.04] px-5 py-4 text-zinc-300">
                    <div className="flex flex-wrap gap-2">
                      <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-100">
                        Reviewing profile details
                      </span>
                      <span className="rounded-full border border-sky-400/20 bg-sky-400/10 px-3 py-1 text-xs font-semibold text-sky-100">
                        Checking recent information
                      </span>
                    </div>
                    <p className="mt-4 text-sm leading-7 text-zinc-300">
                      Thinking through the best answer for you.
                    </p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col items-center justify-center rounded-[30px] border border-dashed border-white/10 bg-white/[0.03] px-6 py-8 text-center sm:px-10">
              <div className="flex h-[4.5rem] w-[4.5rem] items-center justify-center rounded-[24px] bg-amber-300/12 text-amber-100 shadow-[0_0_0_1px_rgba(252,211,77,0.14)]">
                <Bot className="h-9 w-9" />
              </div>
              <p className="mt-6 font-[var(--font-accent)] text-3xl italic text-amber-50">
                Welcome to Adarsh AI
              </p>
              <h3 className="mt-2 text-3xl font-black tracking-tight text-white sm:text-[2.5rem]">
                Ask me anything in plain language.
              </h3>
              <p className="mt-4 max-w-2xl text-sm leading-8 text-zinc-300 sm:text-base">
                Ask me anything about Adarsh's background, projects, and experience, or let me search the live web for current events and recent updates.
              </p>

              <div className="mt-8 flex w-full max-w-2xl flex-wrap justify-center gap-3">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => handlePromptSelection(prompt)}
                    className="rounded-full border border-white/10 bg-white/[0.05] px-4 py-2.5 text-sm text-zinc-200 transition hover:border-amber-200/30 hover:bg-amber-200/10 hover:text-white"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="flex-shrink-0 border-t border-white/10 bg-black/20 px-4 pt-4 pb-6 sm:px-6">
          <form onSubmit={handleSendMessage} className="mx-auto flex max-w-4xl flex-col gap-3 sm:flex-row sm:items-end">
            <label className="flex-1">
              <span className="sr-only">Message</span>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Ask about Adarsh, his resume, projects, or any current topic..."
                className="min-h-[4rem] w-full resize-none rounded-[26px] border border-white/10 bg-white/[0.04] px-4 py-3 text-sm leading-6 text-white outline-none transition placeholder:text-zinc-500 focus:border-amber-200/40 focus:bg-white/[0.06]"
                disabled={isLoading}
                rows={2}
              />
            </label>
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="inline-flex items-center justify-center gap-2 rounded-[26px] border border-amber-300/30 bg-amber-300/15 px-5 py-3 text-sm font-semibold text-amber-50 transition hover:bg-amber-300/20 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              Send Message
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
