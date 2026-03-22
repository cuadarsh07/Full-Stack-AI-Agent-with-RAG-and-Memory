import { ArrowDownToLine, BadgeCheck, BriefcaseBusiness, Sparkles } from 'lucide-react'
import Chat from './Chat'

const highlights = [
  "Ask about Adarsh Kumar's experience, projects, skills, or education.",
  'Get grounded answers from the resume and fresh updates from the live web.',
  'Keep the conversation natural, clear, and easy to restart at any time.',
]

export default function App() {
  return (
    <div className="min-h-screen w-screen overflow-y-auto bg-[radial-gradient(circle_at_top_left,_rgba(255,255,255,0.14),_transparent_22%),radial-gradient(circle_at_bottom_right,_rgba(245,158,11,0.18),_transparent_28%),linear-gradient(160deg,_#f7f1e8_0%,_#e6ddd2_28%,_#17313a_28%,_#0c1b22_100%)] text-stone-100 lg:h-screen lg:overflow-hidden">
      <div className="mx-auto flex min-h-screen w-full max-w-[1600px] flex-col px-3 py-3 sm:px-4 sm:py-4 lg:h-full lg:min-h-0 lg:px-6">
        <div className="grid min-h-full gap-3 lg:h-full lg:min-h-0 lg:grid-cols-[300px_minmax(0,1fr)] lg:gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="relative min-h-0 overflow-y-auto rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,_rgba(250,245,238,0.94)_0%,_rgba(235,226,214,0.86)_100%)] p-5 text-slate-900 shadow-[0_28px_100px_rgba(0,0,0,0.28)] sm:p-6 lg:max-h-full">
            <div className="absolute inset-x-0 top-0 h-32 bg-[radial-gradient(circle_at_top,_rgba(12,27,34,0.10),_transparent_70%)]" />

            <div className="relative flex min-h-full flex-col justify-start gap-4 lg:justify-center">
              <div className="inline-flex w-fit items-center gap-2 rounded-full border border-slate-900/10 bg-white/70 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.28em] text-slate-700">
                <Sparkles className="h-3.5 w-3.5" />
                Personal AI Guide
              </div>

              <div>
                <p className="font-[var(--font-accent)] text-[2rem] italic tracking-tight text-slate-700 sm:text-[2.15rem]">
                  Adarsh AI
                </p>
                <h1 className="mt-2 text-[2rem] font-black leading-tight text-slate-950 sm:text-[2.3rem] xl:text-[2.45rem]">
                  A polished way to explore Adarsh's work and story.
                </h1>
                <p className="mt-3 text-sm leading-6 text-slate-700 sm:text-[14px]">
                  Ask about Adarsh's path, the projects he is proud of, or what he is building toward next in a conversation that feels personal instead of transactional.
                </p>
              </div>

              <div className="rounded-[26px] border border-slate-900/10 bg-white/65 p-4 shadow-[0_18px_50px_rgba(15,23,42,0.08)] backdrop-blur sm:p-5">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 flex-none items-center justify-center rounded-2xl bg-slate-900 text-stone-100">
                    <BadgeCheck className="h-[1.125rem] w-[1.125rem]" />
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
                      Developed by Adarsh Kumar
                    </p>
                    <p className="mt-2 text-sm leading-6 text-slate-700">
                      Full-stack engineer focused on applied AI, product thinking, and practical systems that turn complex workflows into clean user experiences.
                    </p>
                  </div>
                </div>

                <a
                  href="/resume.pdf"
                  className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-stone-50 transition hover:bg-slate-800"
                >
                  <ArrowDownToLine className="h-4 w-4" />
                  Download Resume
                </a>
              </div>

              <div className="space-y-2.5">
                {highlights.map((item) => (
                  <div
                    key={item}
                    className="flex items-start gap-3 rounded-2xl border border-slate-900/8 bg-white/50 px-4 py-3 text-sm leading-5 text-slate-700"
                  >
                    <BriefcaseBusiness className="mt-0.5 h-4 w-4 flex-none text-amber-700" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>

              <div className="pt-1 text-[11px] leading-5 text-slate-500">
                Designed for a calm, human conversation instead of a dashboard full of technical labels.
              </div>
            </div>
          </aside>

          <main className="flex h-full min-h-0 flex-col overflow-hidden">
            <Chat />
          </main>
        </div>
      </div>
    </div>
  )
}
