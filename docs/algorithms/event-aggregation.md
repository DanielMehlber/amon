# Event aggregation and suppression

Detectors run every frame; operators want **one line per incident**, not
thousands of rows.

## From frames to events

For each anomaly ID the pipeline keeps a small state machine:

1. **Idle** — intensity below threshold.
2. **Active** — intensity crossed above threshold; record start time and
   build an intensity timeline.
3. **Cooldown** — intensity dropped, but we wait a short grace period in
   case it flickers back (avoids splitting one incident into many).
4. **Closed** — emit a single event with start, end, duration, peak
   intensity, and the calibrated threshold.

Events shorter than a minimum duration are discarded as glitches.

## Why suppression is needed

One root problem often lights up several detectors:

| Root cause | Side effects |
|---|---|
| Flicker | HUD blink metrics wobble, noise metric rises |
| Noise burst | Contrast metric may twitch |
| HUD resize | Text and position metrics also move |

Without rules, operators see duplicate alarms for the same underlying fault.

## Exclusion hierarchy

Configuration maps a **suppressor** pattern to **target** patterns. While
a suppressor is active, matching targets cannot **open** new events.

Patterns look like paths: `temporal/flicker`, `hud/*/text`. A `*` matches
one or more name segments. When both sides use the same number of `*`
wildcards, captured names must match — so `hud/*/size` suppresses
`hud/*/position` only for the **same** element, not every HUD channel.

## Suppression linger

Some metrics (blink toggle rate) use a **sliding window** and react slowly
when a disturbance ends. A suppressor therefore keeps blocking related
targets for a few seconds **after** it subsides, but only if it was a
**sustained** event — brief spikes do not block unrelated follow-up
anomalies.

## Result

Operators get a concise timeline: one flicker event instead of flicker +
noise + three HUD false positives. The hierarchy is configurable per
deployment without changing detector code.
