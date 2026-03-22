import { ArrowDownToLine, BadgeCheck, BriefcaseBusiness, Sparkles } from 'lucide-react'
import Chat from './Chat'

const NEW_CHAT_EVENT = 'adarsh-ai:new-chat'

const highlights = [
  "Ask about Adarsh Kumar's experience, projects, skills, or education.",
  'Get grounded answers from the resume and fresh updates from the live web.',
  'Keep the conversation natural, clear, and easy to restart at any time.',
]

export default function App() {
  const handleNewChat = () => {
    globalThis.dispatchEvent(new Event(NEW_CHAT_EVENT))
  }

  return (
    <div className="h-[100dvh] w-full flex flex-col md:flex-row overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(255,255,255,0.14),_transparent_22%),radial-gradient(circle_at_bottom_right,_rgba(245,158,11,0.18),_transparent_28%),linear-gradient(160deg,_#f7f1e8_0%,_#e6ddd2_28%,_#17313a_28%,_#0c1b22_100%)] text-stone-100">
      <div className="mx-auto flex h-full min-h-0 w-full max-w-[1440px] flex-1 flex-col gap-3 px-3 py-3 sm:px-4 sm:py-4 md:flex-row md:gap-3 md:px-4 md:py-4 xl:max-w-[1500px]">
        <header className="flex flex-shrink-0 items-center justify-between gap-3 rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,_rgba(250,245,238,0.96)_0%,_rgba(235,226,214,0.9)_100%)] px-4 py-3 text-slate-950 shadow-[0_22px_60px_rgba(0,0,0,0.22)] md:hidden">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-10 w-10 flex-none items-center justify-center rounded-2xl bg-slate-950 text-stone-100">
              <Sparkles className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                Personal AI Guide
              </p>
              <p className="truncate text-lg font-black tracking-tight text-slate-950">Adarsh AI</p>
            </div>
          </div>

          <button
            type="button"
            onClick={handleNewChat}
            className="inline-flex flex-none items-center justify-center rounded-2xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-stone-50 transition hover:bg-slate-800"
          >
            New Chat
          </button>
        </header>

        <aside className="relative hidden min-h-0 overflow-y-auto rounded-[26px] border border-white/10 bg-[linear-gradient(180deg,_rgba(250,245,238,0.94)_0%,_rgba(235,226,214,0.86)_100%)] p-4 text-slate-900 shadow-[0_28px_100px_rgba(0,0,0,0.28)] md:flex md:w-[260px] md:flex-col md:p-5 xl:w-[280px]">
            <div className="absolute inset-x-0 top-0 h-32 bg-[radial-gradient(circle_at_top,_rgba(12,27,34,0.10),_transparent_70%)]" />

            <div className="relative flex min-h-full flex-col justify-start gap-3 lg:justify-center">
              <div className="inline-flex w-fit items-center gap-2 rounded-full border border-slate-900/10 bg-white/70 px-3 py-1 text-[9px] font-semibold uppercase tracking-[0.24em] text-slate-700">
                <Sparkles className="h-3.5 w-3.5" />
                Personal AI Guide
              </div>

              <div>
                <p className="font-[var(--font-accent)] text-[1.7rem] italic tracking-tight text-slate-700 sm:text-[1.85rem] xl:text-[2rem]">
                  Adarsh AI
                </p>
                <h1 className="mt-1.5 text-[1.65rem] font-black leading-tight text-slate-950 sm:text-[1.85rem] xl:text-[2.1rem]">
                  A polished way to explore Adarsh's work and story.
                </h1>
                <p className="mt-2.5 text-[13px] leading-5 text-slate-700 sm:text-[13px]">
                  Ask about Adarsh's path, the projects he is proud of, or what he is building toward next in a conversation that feels personal instead of transactional.
                </p>
              </div>

              <div className="rounded-[22px] border border-slate-900/10 bg-white/65 p-4 shadow-[0_18px_50px_rgba(15,23,42,0.08)] backdrop-blur">
                <div className="flex items-start gap-3">
                  <div className="flex h-9 w-9 flex-none items-center justify-center rounded-xl bg-slate-900 text-stone-100">
                    <BadgeCheck className="h-[1.125rem] w-[1.125rem]" />
                  </div>
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
                      Developed by Adarsh Kumar
                    </p>
                    <p className="mt-2 text-[13px] leading-5 text-slate-700">
                      Full-stack engineer focused on applied AI, product thinking, and practical systems that turn complex workflows into clean user experiences.
                    </p>
                  </div>
                </div>

                <a
                  href="/resume.pdf"
                  className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-stone-50 transition hover:bg-slate-800"
                >
                  <ArrowDownToLine className="h-4 w-4" />
                  Download Resume
                </a>
              </div>

              <div className="space-y-2">
                {highlights.map((item) => (
                  <div
                    key={item}
                    className="flex items-start gap-3 rounded-xl border border-slate-900/8 bg-white/50 px-3.5 py-2.5 text-[13px] leading-5 text-slate-700"
                  >
                    <BriefcaseBusiness className="mt-0.5 h-4 w-4 flex-none text-amber-700" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>

              <div className="pt-1 text-[10px] leading-4 text-slate-500">
                Designed for a calm, human conversation instead of a dashboard full of technical labels.
              </div>
            </div>
          </aside>

        <main className="flex min-h-0 flex-1 overflow-hidden">
          <Chat newChatEventName={NEW_CHAT_EVENT} />
        </main>
      </div>
    </div>
  )
}
