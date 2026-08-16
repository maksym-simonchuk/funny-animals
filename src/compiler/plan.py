"""Ask a local Ollama model to name a compilation and caption its clips.

Two models, both local: a vision one looks at a frame of each clip and says what is in
it, then the text one turns those descriptions into the on-screen ranking.
"""
from __future__ import annotations

import base64
import json
import re
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
    "The category finishes the heading printed on top of every clip, which reads "
    '"TOP 5 <category>", so it has to be a plural noun phrase of at most 3 words. '
    "Each caption is one row of an on-screen ranking: two to four words that read as an "
    "English phrase, never a sentence and never a question, never three nouns in a row. "
    "Write them like meme captions -- the joke has to land in the second it is on screen, "
    "in plain everyday words a stranger scrolling past will get with no explanation. No "
    "wordplay that needs thinking about, no rare words, no inside references. "
    "Never name the animal: the viewer can see what it is, and a row that starts with the "
    'species -- "CHIHUAHUA STRAWBERRY THIEF", "CAT ON A CHAIR" -- is a failure. React to '
    "the clip instead: hand out a verdict, an accusation, a fake job title, a "
    "caught-in-the-act line. The joke still has to land on what the clip shows, so "
    "never invent an animal or an action that is not there. Give every row a different "
    "joke: no shared pattern between rows, no word repeated across them. "
    "Each line is the meme caption printed along the bottom of its own clip, so it belongs "
    "to that one clip alone: at most 8 words, a reaction someone would type as a comment "
    "under it -- what the animal is thinking, what it is being accused of, how it ends. "
    "The same rules hold: plain words, instantly funny, never naming the animal, never "
    "repeating that clip's ranking row. "
    "The title is at most 5 words, the hook at most 8."
)

_LOOK = (
    "Describe this video frame in one short factual clause: what the animal wears or "
    "holds, what it is doing, where it is. Call it \"the animal\" -- never name the "
    "species or the breed. No opinions, no invented details."
)

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
    lines: list[str]  # the meme caption along the bottom of each clip, one per clip


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
            "lines": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": clip_count,
                "maxItems": clip_count,
            },
        },
        "required": ["category", "title", "hook", "captions", "lines"],
    }


_DANGLING = {"a", "an", "the", "in", "of", "and", "with", "to", "for", "on", "at", "its", "his"}


def _row(caption: str) -> str:
    """One ranking row: four words at most, no dangling word, no trailing punctuation.

    The model overruns its own limit on roughly one compilation in five, usually with a
    "X? Then Y" joke. A row that long stops being readable in the second it is on screen,
    so the setup goes and the punchline stays. Cutting at the fourth word can leave the
    row hanging on an article -- "BALL HUGGING IN THE" -- which reads as a broken render.
    """
    words = caption.split()[:4]
    while len(words) > 2 and words[-1].strip(".,!?;:").lower() in _DANGLING:
        words.pop()
    return " ".join(words).rstrip(".,!?;:")


def _line(text: str) -> str:
    """The meme caption along the bottom of one clip: eight words at most.

    It wraps over two lines when it has to, so it can be a whole sentence -- but past
    eight words it covers the clip it is joking about.
    """
    return " ".join(text.split()[:8]).strip()


def _category(text: str) -> str:
    """The tail of the heading, without the "TOP 5" the renderer puts in front of it.

    Told what the finished heading reads like, the model writes the whole heading about
    half the time, which would print as "TOP 5 TOP 5 SWEET TOYS".
    """
    return re.sub(r"^\s*top\s*\d*\s*", "", text, flags=re.IGNORECASE).strip() or "Funny Animals"


def _describe(clips: list[Clip]) -> str:
    lines = []
    for index, clip in enumerate(clips, start=1):
        # the species goes in only when vision is down and it is all there is
        parts = [] if clip.seen else [clip.category]
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


def describe_frame(frame: Path, cfg: "CompilerCfg") -> str:
    """What the local vision model sees in `frame`, as one clause, animal left unnamed.

    The species stays out of it on purpose: told what the animal is, the text model opens
    every ranking row with it ("CHIHUAHUA STRAWBERRY THIEF") however plainly the prompt
    forbids it, and the viewer can see the animal anyway.

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
            "content": _LOOK,
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
        f"exactly {len(clips)} captions and exactly {len(clips)} lines -- one of each per "
        f"clip, in the same order. Both have to be jokes that fit what their clip shows."
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
        category=_category(str(data.get("category") or "")),
        title=str(data.get("title") or "Funny Animals").strip(),
        hook=str(data.get("hook") or "").strip(),
        captions=[_row(str(caption)) for caption in data.get("captions") or []],
        lines=[_line(str(line)) for line in data.get("lines") or []],
    )
    if len(plan.captions) != len(clips):
        raise PlanError(f"model returned {len(plan.captions)} captions for {len(clips)} clips")

    logger.info(f"plan: {plan.category} -- {plan.title}")
    return plan
