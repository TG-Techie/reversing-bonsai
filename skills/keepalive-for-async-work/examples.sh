#!/bin/bash
# Reference snippets for the keepalive-for-async-work skill.
# These are intentionally minimal and self-contained; lift one
# into your agent's runtime and adapt.

set -u

# ---------------------------------------------------------------------------
# Pattern 1: single-shot wait for a known-bounded job.
# Use when you know roughly how long the job takes and just need to
# come back once.
# ---------------------------------------------------------------------------
example_single_shot() {
  nohup ./your_job > work.log 2>&1 &
  local PID=$!
  echo "Spawned PID=$PID; sleeping 60s"

  date -u +%H:%M:%S
  sleep 60
  date -u +%H:%M:%S

  if kill -0 "$PID" 2>/dev/null; then
    echo "Still running; will need a longer wait or a polling loop"
    tail -n 5 work.log
  else
    echo "Complete:"
    tail -n 20 work.log
  fi
}

# ---------------------------------------------------------------------------
# Pattern 2: polling loop with natural completion signal.
# Use when the job duration is uncertain. Combines a foreground
# sleep (for keepalive) with a file-based completion check.
# ---------------------------------------------------------------------------
example_polling_loop() {
  rm -f done.flag
  nohup bash -c './your_job > work.log 2>&1 && touch done.flag' &
  local PID=$!
  echo "Spawned PID=$PID; will poll every 60s up to 30 min"

  local elapsed=0
  local max_wait=1800
  while [ ! -f done.flag ] && [ "$elapsed" -lt "$max_wait" ]; do
    echo "[$(date -u +%H:%M:%S)] still running (elapsed ${elapsed}s); last log:"
    tail -n 3 work.log
    sleep 60
    elapsed=$((elapsed + 60))
  done

  if [ -f done.flag ]; then
    echo "Job completed at $(date -u +%H:%M:%S):"
    tail -n 20 work.log
  else
    echo "Timeout after ${elapsed}s. Job still running (PID=$PID); investigating."
  fi
}

# ---------------------------------------------------------------------------
# Pattern 3: sub-agent wait.
# Same shape as Pattern 2; the "job" is a sub-process / sub-agent
# rather than an inline command. The point is identical: keep the
# parent's turn alive while the child runs.
# ---------------------------------------------------------------------------
example_subagent_wait() {
  nohup ./run_subagent.sh > subagent.log 2>&1 &
  local SUB_PID=$!
  echo "Sub-agent launched (PID=$SUB_PID)"

  while kill -0 "$SUB_PID" 2>/dev/null; do
    echo "[$(date -u +%H:%M:%S)] sub-agent still running; tail:"
    tail -n 3 subagent.log
    date -u +%H:%M:%S && sleep 90 && date -u +%H:%M:%S
  done

  echo "Sub-agent complete:"
  tail -n 20 subagent.log
}

# Uncomment one to run:
# example_single_shot
# example_polling_loop
# example_subagent_wait
