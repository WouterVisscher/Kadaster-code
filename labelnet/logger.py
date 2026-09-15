"""Timestamped progress logging for the labelnet CLI."""

from __future__ import annotations

import datetime as dt


def format_timestamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    """Print a progress line prefixed with the current local timestamp."""
    print(f"[{format_timestamp(dt.datetime.now().astimezone())}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS (hours unbounded)."""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
