"""Ask a local Ollama model to name a compilation and caption its clips.

Two models, both local: a vision one looks at a frame of each clip and says what is in
it, then the text one turns those descriptions into the on-screen ranking.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.config import CompilerCfg

_SYSTEM = (
    "You plan YouTube Shorts compilations of funny animal clips. "
    "Write in English. Punchy and playful, no hashtags, no emoji. "
    "The category is the rubric printed on top of every clip, at most 3 words. "
    "Each caption is one row of an on-screen ranking: a joke about that clip, at most 4 "
    "words. Be funny -- a punchline, a fake job title, a caption the animal would object "
    "to -- but the joke has to land on what the clip actually shows, so never invent an "
    "animal or an action that is not there. "
    "The title is at most 5 words, the hook at most 8."
)

_LOOK = (
    "Describe this video frame in one short factual clause: which animal, what it wears "
    "or holds, what it is doing. Name the species or breed as precisely as you can. "
    "No opinions, no invented details."
)
# YOLO only knows ten classes, so it files a prairie dog under "dog" -- offered as a hint
# the vision model may overrule, never as the answer.
_HINT = ' An object detector labelled the animal "{hint}"; correct it if the frame disagrees.'

_SOUND = (
    "One clip of a funny-animal compilation has no sound of its own, so it has to borrow "
    "the soundtrack of another clip in the same compilation. Pick the one that suits the "
    "silent clip best: match the mood and the energy of what is happening -- a calm scene "
    "wants a calm track, a chaotic one wants a loud track -- rather than matching the "
    "species. Answer with the index of that clip and nothing else."
)


@dataclass(frozen=True)
class Clip:
    """One processed video offered to the model, identified by its database id."""

    video_id: int
    category: str
    tags: list[str]
    seen: str = ""  # what the vision model saw in a frame of this clip


@dataclass(frozen=True)
class Plan:
    category: str
    title: str
    hook: str
    captions: list[str]


class PlanError(RuntimeError):
    """The model is unreachable or answered with something unusable."""


def _schema(clip_count: int) -> dict:
    """JSON schema constraining the reply. Ollama turns this into a decoding grammar,
    so the model cannot emit prose, and cannot return the wrong number of captions."""
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "title": {"type": "string"},
            "hook": {"type": "string"},
            "captions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": clip_count,
                "maxItems": clip_count,
            },
        },
        "required": ["category", "title", "hook", "captions"],
    }


def _describe(clips: list[Clip]) -> str:
    lines = []
    for index, clip in enumerate(clips, start=1):
        parts = [clip.category]
        if clip.tags:
            parts.append("tags: " + ", ".join(clip.tags[:6]))
        if clip.seen:
            parts.append(f'shows: "{clip.seen}"')
        lines.append(f"{index}) " + "; ".join(parts))
    return "\n".join(lines)


def _chat(payload: dict, cfg: "CompilerCfg") -> dict:
    """POST one non-streaming /api/chat call. Raises PlanError on anything unusable."""
    request = urllib.request.Request(
        f"{cfg.host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, OSError) as exc:
        raise PlanError(
            f"cannot reach Ollama at {cfg.host}: {exc}. Start it with `ollama serve`."
        ) from exc
    except json.JSONDecodeError as exc:
        raise PlanError(f"Ollama returned invalid JSON: {exc}") from exc

    if "error" in body:
        raise PlanError(f"Ollama error: {body['error']}")
    return body


def describe_frame(frame: Path, cfg: "CompilerCfg", hint: str = "") -> str:
    """What the local vision model sees in `frame`, as one clause.

    Returns "" when vision is switched off or the model is missing: the plan then falls
    back to the database metadata, which is category and tags and nothing else.
    """
    if not cfg.vision_model:
        return ""

    payload = {
        "model": cfg.vision_model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2},
        "messages": [{
            "role": "user",
            "content": _LOOK + (_HINT.format(hint=hint) if hint else ""),
            "images": [base64.b64encode(frame.read_bytes()).decode("ascii")],
        }],
    }
    try:
        body = _chat(payload, cfg)
    except PlanError as exc:
        logger.warning(f"vision unavailable ({exc}); captions fall back to metadata")
        return ""

    text = (body.get("message") or {}).get("content") or ""
    return " ".join(text.split())[:200]


def pick_sound(target: str, candidates: list[str], cfg: "CompilerCfg") -> int:
    """Index of the clip whose audio best suits a silent one, judged by the local model.

    Falls back to 0 whenever the model is unreachable or answers out of range: a
    mismatched track still beats a hole in the sound.
    """
    if len(candidates) < 2 or not target:
        return 0

    options = "\n".join(f"{index}) {text}" for index, text in enumerate(candidates))
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": {
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": _SOUND},
            {"role": "user", "content": f'Silent clip: "{target}"\n\nSoundtracks:\n{options}'},
        ],
    }
    try:
        body = _chat(payload, cfg)
        chosen = json.loads((body.get("message") or {}).get("content") or "{}")["index"]
    except (PlanError, KeyError, json.JSONDecodeError) as exc:
        logger.warning(f"sound match unavailable ({exc}); taking the first track")
        return 0

    # the schema pins the type but not the range, so the answer still needs checking
    return chosen if isinstance(chosen, int) and 0 <= chosen < len(candidates) else 0


def make_plan(clips: list[Clip], cfg: "CompilerCfg") -> Plan:
    """Return the compilation plan for `clips`, in order. Raises PlanError on failure."""
    if not clips:
        raise PlanError("no clips to plan")

    prompt = (
        f"Clips:\n{_describe(clips)}\n\n"
        f"Invent the compilation category, an on-screen title, a first-second hook, "
        f"and exactly {len(clips)} captions -- one per clip, in the same order. "
        f"Each caption must be a joke that fits what its clip shows."
    )
    payload = {
        "model": cfg.model,
        "stream": False,
        # qwen3 and friends emit a reasoning block by default; we only want the JSON.
        "think": False,
        "format": _schema(len(clips)),
        "options": {"temperature": cfg.temperature},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    body = _chat(payload, cfg)

    content = (body.get("message") or {}).get("content") or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlanError(f"model did not return JSON: {content[:200]}") from exc

    plan = Plan(
        category=str(data.get("category") or "Funny Animals").strip(),
        title=str(data.get("title") or "Funny Animals").strip(),
        hook=str(data.get("hook") or "").strip(),
        captions=[str(c).strip() for c in data.get("captions") or []],
    )
    if len(plan.captions) != len(clips):
        raise PlanError(f"model returned {len(plan.captions)} captions for {len(clips)} clips")

    logger.info(f"plan: {plan.category} -- {plan.title}")
    return plan
