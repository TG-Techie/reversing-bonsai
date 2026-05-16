---
name: keepalive-for-async-work
description: Hold an agent's turn open while a long-running background process or sub-agent runs, so the work doesn't finish silently in a suspended session. Use when you've spawned a non-trivial async task (download, training run, sub-agent, external API call) and need to be present when it completes.
---

# Keepalive for async work

When you spawn long-running async work (a background process, a tool
call to a downstream service, a sub-agent), many agent harnesses will
**suspend your turn the moment there's no foreground activity**. The
work itself continues; you just never see its completion. Result:
silent failures or silent successes that look the same.

This skill encodes the defensive pattern: spawn, hold the turn open
with a bounded foreground wait, then check. It's framework-agnostic
— anywhere you can run shell commands or your agent has a "sleep"
primitive, it applies.

## The why (one paragraph)

A turn-based agent is alive only while it's doing visible work. A
background process is invisible to the harness's "is the turn still
active?" check. So if you do `bg_task &` and immediately end your
turn, the harness can pause the whole session; the background task
keeps running and finishes alone, and by the time you (or the user)
look, the result is in a log file the harness never told you about.
The fix is to give the harness *something to observe* — a foreground
sleep — for long enough that the background task finishes inside it.
Then check separately.

## The recipe (shell)

```bash
# 1. Spawn as a TRUE background process.
#    If your harness has a "run in background" wrapper,
#    prefer raw `nohup ... &` instead — wrappers have a
#    documented tendency to lose work across suspends.
nohup ./long_running_command > work.log 2>&1 &
JOB_PID=$!
echo "Spawned PID=$JOB_PID; sleeping 60s before first check"

# 2. Hold the turn with a bracketed foreground sleep.
#    Bracket with timestamps so the gap is visible in logs.
date -u +%H:%M:%S
sleep 60
date -u +%H:%M:%S

# 3. Check in a SEPARATE call (clean log boundaries).
if kill -0 "$JOB_PID" 2>/dev/null; then
  echo "Still running; tail of progress:"
  tail -n 5 work.log
  # Loop step 2 + 3 until done
else
  echo "Complete. Final output:"
  tail -n 20 work.log
fi
```

## Sizing the sleep

Match the sleep duration to expected work time, not to gut feel. The
ladder that works:

- **30 s** — fast operations (HTTP fetch, small file download, quick
  shell tool).
- **60 s** — moderate work (small model load, modest computation).
- **90 s** — medium work (one phase of a training step, file
  preprocessing).
- **180 s** — heavy work (a single QAT iteration on a large model, a
  full benchmark run).
- **300 s** — last resort. Past this duration, prefer multiple shorter
  polls so the user can see progress.

Multiple short polls beat one long sleep for log readability and
recovery: if the agent dies on the wrong side of a 300s sleep you lose
visibility for 5 minutes; if it dies between 60s sleeps you lose 60s.

## Announce before sleeping

A bare `sleep 60` in the log is indistinguishable from the agent
hanging. Print one sentence of intent **before** the sleep, so:
- the human watching the transcript (often on mobile) knows what's
  happening,
- future-you reading the log months later can tell.

Example: `"Sleeping 60s while the C4 download finishes (~50MB)"`.

## Don't make sleep the only liveness

A single foreground `sleep` is a single point of failure. If the
harness kills it for any reason, you're stranded.

Mitigations (use one or more):

- **Status file polling**: have the background process touch a
  `done.flag` file when it finishes; the parent checks for that file's
  existence on each wake.
- **Webhook / external signal**: a remote service can ping you when
  the work is done; you wake on the inbound message rather than the
  sleep.
- **Follow-up poll**: after a sleep, do a quick `tail`, `kill -0`, or
  job-status check so you have BOTH the sleep AND an explicit check
  signal.

## Conventions

- **One announcement per sleep batch**, not per sleep call. The user
  doesn't need a line every 60 s; they need a line per intentional
  wait.
- **Sub-agents don't keepalive.** Sub-agents are short-lived checks
  that terminate when done. Keepalive applies to the top-level agent
  — the one whose turn the harness might suspend.
- **After 2× expected work-time, investigate**. If a job has run twice
  as long as you thought it would, it's stuck or the estimate was
  wrong. Don't sit on a stuck job; the grant is non-renewable.
- **Use natural completion signals where possible**: file existence,
  process state, lock release. Time-based "wait long enough" is the
  fallback, not the default.

## When this pattern does NOT apply

- **Waiting for user input.** Different problem entirely — end your
  turn, wait for the next user message. Don't sleep.
- **Tight inner loops in your code** (millisecond scale). Use proper
  async / await, not sleep-poll.
- **Sub-agent execution.** Sub-agents terminate when done; they don't
  need keepalive. The parent agent does.

## Platform-specific notes

These are observations from one agent platform; the pattern itself
is portable but the gotchas may differ:

- **Claude Code (CLI and web)**: the Bash tool's `run_in_background:
  true` option has been observed to lose work across session suspends.
  Prefer raw `nohup ... &` for any non-trivial async job. The web
  variant additionally may block "long leading sleeps" by policy —
  workaround: use a `Monitor` tool with an `until <condition>` loop,
  or chain short bracketed sleeps.

- **Cursor / Aider / similar IDE agents**: usually keep the foreground
  alive while shell commands run, but background processes started
  via `&` may not be tracked between turns. Same pattern applies.

- **API-based agents (raw Anthropic / OpenAI SDK)**: you control the
  turn lifecycle yourself. The pattern collapses to "your code
  loop polls until the job is done." But the discipline of bracketing
  with timestamps and announcing intent still pays off — your own
  logs are easier to read.

- **Custom agent frameworks (CrewAI, AutoGPT, LangGraph, etc.)**:
  check whether the framework has a built-in "wait for async
  primitive" first. If yes, use it. If not, this pattern is the
  fallback.

## Concrete example: waiting on a downstream sub-agent

```bash
# Spawn a sub-agent / external job
nohup ./run_subagent.sh > subagent.log 2>&1 &
SUB_PID=$!
echo "Launched sub-agent (PID=$SUB_PID); first poll in 90s"

# Bracketed wait
date -u +%H:%M:%S && sleep 90 && date -u +%H:%M:%S

# Check; loop if not done
while kill -0 "$SUB_PID" 2>/dev/null; do
  echo "Sub-agent still running; tail:"
  tail -n 3 subagent.log
  echo "Waiting another 60s"
  date -u +%H:%M:%S && sleep 60 && date -u +%H:%M:%S
done

# Done
echo "Sub-agent complete. Final output:"
cat subagent.log
```

## What this skill does NOT cover

- Choosing *what* to spawn (that's the work itself).
- Recovering from a failed background job (separate skill: retry-
  with-backoff).
- Coordinating multiple parallel background jobs (separate problem:
  job orchestration).
- Distinguishing "still running" from "hung" (use natural completion
  signals + 2× timeout heuristic above).

## Provenance

This pattern was developed through hours of debugging a Claude Code
web session where background processes were silently completing
inside suspended turns. The specific sizing (30 / 60 / 90 / 180 /
300 s ladder) is empirical: shorter polls felt jittery and produced
unreadable logs; longer polls lost visibility on failed jobs. The
"don't make sleep the only liveness" rule came from one bad sleep
that the harness killed for unrelated reasons, leaving the parent
stranded.

The discipline is generic. The platform-specific notes above are
where it bites in practice; expect your platform to have its own
analogous gotchas.
