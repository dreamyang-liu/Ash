"""Native agent events -> unified journal events.

Each module here is a pure mapping table: no I/O, no subprocess, no SDK import.
That keeps them unit-testable against recorded fixtures and makes them the only
place that has to change when an upstream CLI bumps its event schema (see
contracts/ for the drift checks).

Contract for every ``normalize(native) -> list[tuple[str, dict]]``:
- Return zero or more ``(event_type, payload)`` pairs, in order.
- Never raise on unknown input: fall back to ``(raw_event(slot), {...})`` so
  the journal keeps a verbatim copy. Silent drops are the one unacceptable
  failure mode -- they corrupt trajectories without any error surfacing.
"""
