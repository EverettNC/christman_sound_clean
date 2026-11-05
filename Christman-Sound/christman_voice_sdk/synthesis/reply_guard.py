# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
Lines a being never says to a client: a word list the being reads.

WHAT CHANGED AND WHY (2026-11-04)
---------------------------------
Everett, 2026-11-04: "we can have this knowledge in the sound package, but it
can't be making those decisions. It's up to the being to make those
decisions, not something that just creates a voice."

Before this, check_being_reply raised ForbiddenReply when a being's reply
carried one of these lines, and nothing was spoken; with silent_alarm=True it
also refused any reply that announced help. That was the sound package
deciding what a being may say. Now the lists stay here as knowledge and
nothing here refuses: forbidden_lines(reply, danger=False) returns the names
of the lines it finds, and the being decides what to say. ForbiddenReply and
the silent_alarm parameter are gone. check_being_reply never raises.

The lists
---------
FORBIDDEN_REPLY_LINES. Everett, 2026-10-29 (ruling E): "you're not okay" is a
statement said at someone, and as a hard rule no being ever says "you're not
okay" or "calm down" to a client.

ANNOUNCEMENT_LINES. Ruling I, 2026-10-29: lines that announce help, a call,
or someone coming. Everett's reason: saying it out loud can get the client
hurt worse and the device taken. They are checked only when the caller
passes danger=True, the being's own word that it has a danger reading in
hand (the gate's SafetyResult.danger).

Both lists are read on the being's own reply only. Nothing here reads,
changes or holds the client's words.

Each line is matched as words, in any case, with straight or curly
apostrophes, wherever it sits in the reply. "if you're not okay WITH that,
we can stop" is a question about consent, not a statement about the client,
and is not named, as "not okay with it" is left alone everywhere else in
this package.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#: (name, pattern). Read on the being's reply only (ruling E).
FORBIDDEN_REPLY_LINES: Tuple[Tuple[str, str], ...] = (
    ("you're not okay",
     r"\byou(?:'re|re|\s+are)\s+not\s+(?:really\s+|very\s+)?"
     r"(?:okay|ok|alright|all\s+right)\b(?!\s+with\b)"),
    ("calm down", r"\bcalm\s+(?:yourself\s+)?down\b"),
)

#: (name, pattern). Lines that announce help, a call, or someone coming
#: (ruling I). Read on the being's reply only, and only when danger=True.
ANNOUNCEMENT_LINES: Tuple[Tuple[str, str], ...] = (
    ("help is coming", r"\bhelp\s+(?:is\s+)?(?:on\s+(?:the|its|it's)\s+way|coming)\b"),
    ("I am getting help",
     r"\b(?:i'?m|i\s+am|i'?ve|i\s+have|i\s+will|i'?ll|we'?re|we\s+are|we'?ve)\s+(?:just\s+)?"
     r"(?:getting|got|called|calling|call|contacted|contacting|alerted|alerting|"
     r"notified|notifying|paged|paging|sent\s+for|sending\s+for|get)\b"),
    ("someone is coming",
     r"\b(?:someone|somebody|a\s+person|people|the\s+police|police|911|an\s+ambulance|help)\s+"
     r"(?:is|are|will\s+be|has\s+been)\s+(?:coming|on\s+(?:the|their|its|his|her)\s+way|called|notified|alerted)\b"),
    ("calling for help",
     r"\b(?:calling|getting|alerting)\s+(?:help|someone|somebody|the\s+police|911|an\s+ambulance)\b"),
)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'"})


def forbidden_lines(reply: Optional[str], danger: bool = False) -> List[str]:
    """
    The names of the lines found in a being's reply, in list order. Reads
    only: nothing is raised, nothing is changed, nothing is spoken or held.

    danger: True when the being has a danger reading in hand and wants the
    announcement lines (ANNOUNCEMENT_LINES) checked as well. False checks
    FORBIDDEN_REPLY_LINES only.

    Never call this on the client's own words.
    """
    lowered = (reply or "").translate(_APOSTROPHES).lower()
    lines = FORBIDDEN_REPLY_LINES + (ANNOUNCEMENT_LINES if danger else ())
    return [name for name, pattern in lines if re.search(pattern, lowered)]


def check_being_reply(reply: Optional[str], danger: bool = False) -> str:
    """
    Return the being's reply unchanged. Never raises.

    The lines forbidden_lines finds are logged by name at WARNING, so the
    reading is on the record. Whether to say the reply, change it, or say
    something else is the being's decision; forbidden_lines is how the being
    reads the names. Kept under this name for callers that called it before
    2026-11-04, when it raised ForbiddenReply instead.

    Never call this on the client's own words.
    """
    found = forbidden_lines(reply, danger=danger)
    if found:
        logger.warning("The being's reply carries %s (read only; the being decides).",
                       ", ".join(repr(name) for name in found))
    return reply or ""


__all__ = ["FORBIDDEN_REPLY_LINES", "ANNOUNCEMENT_LINES",
           "check_being_reply", "forbidden_lines"]
