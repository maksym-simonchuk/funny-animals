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
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.config import CompilerCfg

_SYSTEM = (
    "You plan YouTube Shorts compilations of funny animal clips. "
    "Write in English. Punchy and playful, no hashtags, no emoji. "
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
    "No alliteration and no rhyme: three words starting on the same letter read as a "
    'tongue twister, not a joke -- "BOW BURDENED BADDY" is the failure. '
    "The title is at most 5 words, the hook at most 8."
)

_LOOK = (
    "Look at this video frame and answer with two fields. \"animal\" is what the animal is, "
    "one or two words, the breed if you can tell it. \"scene\" is one short factual clause "
    "about what it wears or holds, what it is doing, where it is -- and in that clause you "
    "call it \"the animal\", never naming it. No opinions, no invented details."
)
_LOOK_SCHEMA = {
    "type": "object",
    "properties": {"animal": {"type": "string"}, "scene": {"type": "string"}},
    "required": ["animal", "scene"],
}

# the enum is the decoding grammar: an answer other than these two is impossible
_YES_NO = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
}

_TAGS = [
    "stealing food", "eating", "drinking", "sleeping", "wearing clothes", "in water",
    "climbing", "falling over", "yelling", "riding something", "playing with a toy",
    "getting a bath", "begging for food", "standing on hind legs", "chasing", "hiding",
    "watching something", "being held", "escaping", "making a mess",
]

# what each tag reads as over the ranking when the model cannot beat it: three tries at
# "TOP 5 <something>" end in "WEARING CLOTHES" often enough to be worth writing out once
_HEADINGS = {
    "stealing food": "Snack Bandits", "eating": "Hungry Professionals",
    "drinking": "Thirsty Rascals", "sleeping": "Professional Nappers",
    "wearing clothes": "Fashion Icons", "in water": "Pool Party Regulars",
    "climbing": "Wall Climbers", "falling over": "Gravity Victims",
    "yelling": "Loud Complainers", "riding something": "Backseat Passengers",
    "playing with a toy": "Toy Destroyers", "getting a bath": "Bath Time Haters",
    "begging for food": "Shameless Beggars", "standing on hind legs": "Tall Boys",
    "chasing": "Hot Pursuers", "hiding": "Master Hiders",
    "watching something": "Nosy Neighbours", "being held": "Held Hostages",
    "escaping": "Great Escapers", "making a mess": "Chaos Machines",
    # the prop buckets get their own, and they read better than the action ones: the thing
    # in the frame is concrete, so the heading can be too
    "fruit": "Fruit Feasts", "junk food": "Snack Raiders", "a drink": "Bar Regulars",
    "a box": "Box Dwellers", "a mirror": "Mirror Fighters", "a screen": "Screen Addicts",
    "clothes": "Fashion Victims", "water": "Pool Party Regulars", "a bed": "Bed Hogs",
    "a toy": "Toy Destroyers", "furniture": "Furniture Wreckers", "a car": "Backseat Drivers",
    "snow": "Snow Goblins", "another animal": "Odd Couples", "a baby": "Tiny Babysitters",
}

# the wider net: several labels one heading can still say honestly. The key is the phrase
# every candidate is asked about (`fits_tag`), so a clip that slipped into a member label
# does not ride into the family on it; the value is the labels and the heading to fall
# back on. Used only after single labels run dry -- "TOP 5 SNACK BANDITS" over five
# thieves beats "TOP 5 FOOD CRIMINALS" over five animals merely near food
_FAMILIES: dict[str, tuple[set[str], str]] = {
    "sneaking or stealing": ({"stealing food", "hiding", "escaping"}, "Sneaky Operators"),
    "eating or drinking": (
        {"eating", "drinking", "stealing food", "begging for food", "fruit", "junk food",
         "a drink"}, "Hungry Legends"),
    # the phrase is what the gate reads, so it stays short: asked whether a clip fits
    # "acting like a person -- dressed up, standing upright, riding or watching a screen",
    # the model said yes to one clip in five and the family never filled
    "acting like a human": (
        {"wearing clothes", "standing on hind legs", "riding something", "clothes",
         "a screen", "a mirror"}, "Human Impersonators"),
    "in or near water": ({"in water", "getting a bath", "water"}, "Splash Squad"),
    "asleep or lounging": (
        {"sleeping", "a bed", "furniture", "being held"}, "Professional Loungers"),
    "causing chaos": (
        {"making a mess", "falling over", "chasing", "yelling"}, "Chaos Machines"),
    "a toy or a box in the frame": (
        {"playing with a toy", "a toy", "a box"}, "Playtime Professionals"),
}

_TAG = (
    "Here is one clip from a pool of funny animal clips. Answer with the one label from "
    "the list that matches what the animal is doing in it. If none of them really "
    'matches, answer "other": the label decides which compilation the clip lands in, and '
    "a clip that does not belong makes the heading over it a lie. Match the label to what "
    "is odd in the clip, not to a word in it -- an animal simply standing on four legs is "
    'not "standing on hind legs", and a bowl in the corner is not "eating".'
)

_LINE = (
    "Write the meme caption printed along the bottom of one clip of a funny-animal "
    "compilation: at most 8 words, the reaction someone would type as a comment under it "
    "-- what the animal is thinking, what it is being accused of, how this ends. Plain "
    "everyday words, funny the second it is read, no hashtags, no emoji, no question, and "
    "no alliteration -- words picked for starting on the same letter stop being funny. "
    "Never name the animal, the viewer can see it. Joke about what this one clip shows "
    "and nothing else -- invent no animal and no action that is not in it -- and never "
    "repeat the row already printed over it."
)

# the second axis: not what the animal does but what it does it to. "fruit" and "a mirror"
# group clips that no action label ever puts together, and a pool this size runs out of
# full action buckets around the seventh compilation
_PROPS = [
    "fruit", "junk food", "a drink", "a box", "a mirror", "a screen", "clothes", "water",
    "a bed", "a toy", "furniture", "a car", "snow", "another animal", "a baby",
]

_PROP = (
    "Here is one clip from a pool of funny animal clips. Answer with the one thing from the "
    "list that is in the frame with the animal and is part of why the clip is funny. If none "
    'of them is in it, or the thing is only scenery in the corner, answer "other": the label '
    "decides which compilation the clip lands in, and a clip that does not belong makes the "
    "heading over it a lie."
)

_THEME = (
    "These clips are going into one YouTube Shorts compilation. They all show the same "
    'thing -- "{tag}". Name them: the heading will read "TOP {size} <your answer>", so '
    "answer with two or three words, a plural noun with something in front of it, the "
    "kind of label a meme would hang on all of them at once. Plain words a stranger gets "
    'at a glance, in the shape of "Snack Bandits", "Gravity Victims", "Professional '
    'Nappers", "Fruit Feasts": the last word is who or what they ARE, plural because the '
    "heading counts five of them, and it carries the joke on its own. Not a verb phrase, "
    "and never an empty word in that last slot -- "
    '"ACTS", "FUN", "MOMENTS", "VIBES", "ANTICS", "PETS" name nothing. Do not name any '
    "animal in them. All five have to fit under it: a detail that shows up in only one of "
    "them -- a watermelon, a mirror, a blanket -- makes the heading a lie over the other "
    "four, so name what they have in common and nothing else."
)

_FITS = (
    "Here is one clip and the label the compilation it is about to join was built on. Answer "
    "yes if the thing the label names -- the object, the action, the species -- is in this "
    "clip. A more exact word for the same thing counts: a parrot is a bird, a dachshund is a "
    "dog. Answer no when that thing is not there at all: a groundhog is not a dog, a carrot "
    "is not a toy. The heading over the set says the label about every clip in it."
)

_TRUE = (
    "Here is one clip of a compilation and the heading that will be printed over it. Answer "
    "yes only if the heading is true of this clip. An object or an action named in the "
    "heading that is not in the clip makes the heading a lie about it."
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
    seen: str = ""  # what the vision model saw in a frame of this clip, animal unnamed
    animal: str = ""  # what the vision model called the animal; only the heading gets it
    tag: str = ""  # one of _TAGS: what the clip is of, the thing a compilation groups on
    prop: str = ""  # one of _PROPS: what is in frame with it, the other thing to group on
    text: bool = False  # the clip carries its own burned-in caption, so ours collides


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
            "title": {"type": "string"},
            "hook": {"type": "string"},
            "captions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": clip_count,
                "maxItems": clip_count,
            },
        },
        "required": ["title", "hook", "captions"],
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
    # the full stop is invisible in caps and only shows up on the captions that have one;
    # a comma or a colon is where the eighth word cut a longer sentence in half, and it
    # prints as one -- "THE ANIMAL IS PARTIALLY HIDDEN BEHIND THE BABY,"
    return " ".join(text.split()[:8]).strip().rstrip(".,;:")


def _category(text: str) -> str:
    """The tail of the heading, without the "TOP 5" the renderer puts in front of it.

    Told what the finished heading reads like, the model writes the whole heading about
    half the time, which would print as "TOP 5 TOP 5 SWEET TOYS".
    """
    return re.sub(r"^\s*top\s*\d*\s*", "", text, flags=re.IGNORECASE).strip() or "Funny Animals"


def _hide(scene: str, animal: str) -> str:
    """`scene` with the species swapped for "the animal".

    The vision model names it anyway about a third of the time ("the ground squirrel is
    being held"), and one species word in the prompt is enough for qwen3 to open every
    ranking row with it.
    """
    words = animal.split()
    for name in filter(None, [" ".join(words), words[-1] if len(words) > 1 else ""]):
        # "the meerkats are standing" is the same leak as "the meerkat is standing"
        pattern = (r"\b(?:the |a |an )?"
                   + r"\s+".join(re.escape(word) for word in name.split()) + r"s?\b")
        scene = re.sub(pattern, "the animal", scene, flags=re.IGNORECASE)
    return scene


def _describe(clips: list[Clip]) -> str:
    lines = []
    for index, clip in enumerate(clips, start=1):
        # the species goes in only when vision is down and it is all there is
        parts = [] if clip.seen else [clip.category]
        if clip.tags:
            parts.append("tags: " + ", ".join(clip.tags[:6]))
        if clip.seen:
            parts.append(f'shows: "{_hide(clip.seen, clip.animal or clip.category)}"')
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


def unload(model: str, cfg: "CompilerCfg") -> None:
    """Drop `model` out of Ollama's memory. Best effort -- a failure changes nothing."""
    if not model:
        return
    try:
        _chat({"model": model, "keep_alive": 0, "messages": []}, cfg)
    except PlanError as exc:
        logger.debug(f"could not unload {model}: {exc}")


def describe_frame(frame: Path, cfg: "CompilerCfg") -> tuple[str, str]:
    """What the local vision model sees in `frame`, as (animal, scene).

    The two are kept apart on purpose. The scene leaves the animal unnamed, because the
    text model that writes the ranking opens every row with the species it is given
    ("CHIHUAHUA STRAWBERRY THIEF") however plainly the prompt forbids it. The name still
    has to exist, though: without it the heading comes out as "TOP 5 SWEET THIEVES",
    true of nothing in particular, when it should say which animals are in the set.

    Returns ("", "") when vision is switched off or the model is missing: the plan then
    falls back to the database metadata, which is category and tags and nothing else.
    """
    if not cfg.vision_model:
        return "", ""

    payload = {
        "model": cfg.vision_model,
        "stream": False,
        "think": False,
        "format": _LOOK_SCHEMA,
        "options": {"temperature": 0.2},
        "messages": [{
            "role": "user",
            "content": _LOOK,
            "images": [base64.b64encode(frame.read_bytes()).decode("ascii")],
        }],
    }
    try:
        body = _chat(payload, cfg)
        data = json.loads((body.get("message") or {}).get("content") or "{}")
    except (PlanError, json.JSONDecodeError) as exc:
        logger.warning(f"vision unavailable ({exc}); captions fall back to metadata")
        return "", ""

    animal = " ".join(str(data.get("animal") or "").split())[:40]
    return animal, " ".join(str(data.get("scene") or "").split())[:200]


_TEXT = ("Look at the picture. Is there large text burned into it -- a caption, a title, a "
         "subtitle or a numbered list? A small username, logo or channel handle does not "
         "count. Answer yes or no.")


def has_text(frame: Path, cfg: "CompilerCfg") -> bool:
    """Does the clip already carry its own caption?

    A reel that came with a "Top 10 Funniest Cats" list burned into the picture puts two
    rankings on the screen at once and reads as broken. The small @handle every reel
    carries is fine, so the question says so -- asked this way the model got all eight
    probe clips right.
    """
    if not cfg.vision_model:
        return False

    payload = {
        "model": cfg.vision_model,
        "stream": False,
        "think": False,
        "format": _YES_NO,
        "options": {"temperature": 0.0},
        "messages": [{
            "role": "user",
            "content": _TEXT,
            "images": [base64.b64encode(frame.read_bytes()).decode("ascii")],
        }],
    }
    try:
        body = _chat(payload, cfg)
        answer = json.loads((body.get("message") or {}).get("content") or "{}").get("answer")
    except (PlanError, json.JSONDecodeError) as exc:
        logger.warning(f"burned-text check unavailable ({exc}); the clip is kept")
        return False
    return answer == "yes"


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


def tag_clip(scene: str, cfg: "CompilerCfg") -> str:
    """Which of `_TAGS` the clip is an instance of, or "other" -- what the animal is doing."""
    return _label(scene, _TAG, _TAGS, cfg)


def tag_prop(scene: str, cfg: "CompilerCfg") -> str:
    """Which of `_PROPS` is in the frame with the animal, or "other" -- what it does it to.

    The second axis a set can honestly share. Five animals doing five different things to
    fruit is as true a compilation as five animals all stealing, and with a pool this size
    it is often the only bucket left that fills.
    """
    return _label(scene, _PROP, _PROPS, cfg)


def fits_tag(scene: str, label: str, cfg: "CompilerCfg") -> bool:
    """Is this clip really an instance of `label`?

    The tagging pass is one shot at temperature 0 and it misfires: a cockatoo facing a cat
    came back as "a toy". Asked about one clip and one label at a time, with the answer
    constrained to yes/no, the same model catches its own miss.

    The question is asked strictly. Told a clip "does not have to be the perfect example",
    the model waved a panda eating a carrot into "TOP 5 MORE TOY DESTROYERS" and a groundhog
    into "TOP 5 CERTIFIED DOGS" -- the detector class is wrong often enough that this gate
    is the only thing standing between it and the heading.
    """
    return _yes_no(_FITS, f'Label: "{label}"\n\nClip: "{scene}"', cfg)


def heading_holds(theme: str, clips: list["Clip"], cfg: "CompilerCfg") -> bool:
    """Is `theme` true of all five clips? The last gate before it goes on the video.

    Asked one clip at a time. With all five in one prompt the model answers for the list as
    a whole and waves through a heading that fits some of it: "TOP 5 SKY WATCHERS" passed on
    a set where one animal looked up and the other four looked at a table, a window and a
    cake. One clip per question leaves it nothing to average over.
    """
    return all(
        _yes_no(_TRUE, f'Heading: "{theme}"\n\nClip: {_hide(clip.seen, clip.animal or clip.category)}', cfg)
        for clip in clips
    )


def _yes_no(prompt: str, question: str, cfg: "CompilerCfg") -> bool:
    """A yes/no out of the model, enum-constrained. Unreachable model answers yes: the
    checks are there to catch the model, not to block the build when it is down."""
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": _YES_NO,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    }
    try:
        body = _chat(payload, cfg)
        return json.loads((body.get("message") or {}).get("content") or "{}").get("answer") == "yes"
    except (PlanError, json.JSONDecodeError) as exc:
        logger.warning(f"check unavailable ({exc}); letting it through")
        return True


def _label(scene: str, prompt: str, options: list[str], cfg: "CompilerCfg") -> str:
    """The one label out of `options` that fits `scene`, or "other".

    A compilation is a bucket of clips that share one of these, so the label has to be
    picked per clip and cannot be a matter of taste: the enum in the schema becomes the
    decoding grammar, and the model can answer nothing else.
    """
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": {
            "type": "object",
            "properties": {"tag": {"type": "string", "enum": [*options, "other"]}},
            "required": ["tag"],
        },
        "options": {"temperature": 0.0},  # a label, not a joke: the same clip gets the same one
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f'Labels: {", ".join(options)}\n\nClip: "{scene}"'},
        ],
    }
    try:
        body = _chat(payload, cfg)
        tag = json.loads((body.get("message") or {}).get("content") or "{}").get("tag")
    except (PlanError, json.JSONDecodeError) as exc:
        logger.warning(f"tagging unavailable ({exc}); the clip joins no theme")
        return "other"
    return tag if tag in options else "other"


def group_clips(clips: list[Clip], size: int, cfg: "CompilerCfg",
                done: set[str] | None = None) -> tuple[list[int], str]:
    """Pick `size` clips that share a tag, as (indices, heading).

    A compilation whose clips were chosen by recency has nothing true to say about
    itself, and the heading comes out as "TOP 5 SWEET THIEVES" over five animals that are
    not stealing anything. The set is a bucket of one tag, so the heading is true of every
    clip in it by construction -- asking the model to both pick the five and name what
    they share put clips in it that only half fitted.

    The fullest bucket goes first. Across a batch that keeps the themes moving: each short
    empties the bucket it used, and the next short has to look elsewhere.

    ``done`` is read and extended in place: a label a short in this batch already used is
    skipped even when it still has clips, or the pool's biggest bucket names two shorts in
    a row and both get the same fallback heading over different clips.
    """
    done = set() if done is None else done
    labels: dict[str, list[int]] = {}
    for index, clip in enumerate(clips):
        # two axes in one pile: what the animal does and what it does it to. Both are
        # things all five clips of a bucket genuinely share, and the fuller one wins
        for label in (clip.tag, clip.prop):
            if label and label != "other":
                labels.setdefault(label, []).append(index)

    tag, picked = _fullest(clips, labels, size, cfg, done)
    if not picked:
        # a single label is the funniest set and runs out first; a family is several
        # labels one heading still covers honestly, so the batch keeps having themes
        # after "stealing food" alone is down to two clips
        families = {
            phrase: [index for index, clip in enumerate(clips)
                     if members & {clip.tag, clip.prop}]
            for phrase, (members, _) in _FAMILIES.items()
        }
        tag, picked = _fullest(clips, families, size, cfg, done)
        if picked:
            logger.info(f"theme: {tag} (family) -- clips {[i + 1 for i in picked]}")
            done.add(tag)
            return picked, name_theme(tag, [clips[i] for i in picked], size, cfg,
                                      fallback=_FAMILIES[tag][1], done=done)
    if not picked:
        # late in a batch the actions run dry; the species is the last thing a set can
        # honestly share -- "TOP 5 CERTIFIED CATS" is true of five cats however random.
        # Both the vision word ("chipmunk") and the detector class ("dog") go in: the
        # detector only knows nine animals, so its buckets are the ones deep enough to
        # still fill a short at the end of a batch -- and wrong often enough that the
        # gate has to read every candidate before it counts
        species: dict[str, list[int]] = {}
        for index, clip in enumerate(clips):
            for label in {clip.animal.lower(), clip.category.lower()}:
                if label:
                    species.setdefault(label, []).append(index)
        tag, picked = _fullest(clips, species, size, cfg, done)
        if picked:
            logger.info(f"theme: {tag} (species) -- clips {[i + 1 for i in picked]}")
            done.add(tag)
            return picked, name_theme(tag, [clips[i] for i in picked], size, cfg,
                                      fallback=f"Certified {tag.title()}s", done=done)
    if not picked:
        # nineteen shorts need more themes than the pool has labels. Once every label is
        # spoken for, a second short off the same one over different clips beats a run of
        # shorts all called "Funny Animals" -- `done` holds the headings too, so this one
        # gets its own name
        tag, picked = _fullest(clips, labels, size, cfg, set())
        if picked:
            logger.info(f"theme: {tag} (again) -- clips {[i + 1 for i in picked]}")
            return picked, name_theme(tag, [clips[i] for i in picked], size, cfg, done=done)
    if not picked:
        # nothing groupable left -- an honest generic heading beats a themed lie
        logger.info(f"nothing has {size} clips left to share; falling back to the newest ones")
        return list(range(min(size, len(clips)))), "Funny Animals"

    logger.info(f"theme: {tag} -- clips {[index + 1 for index in picked]}")
    done.add(tag)
    return picked, name_theme(tag, [clips[index] for index in picked], size, cfg, done=done)


def _fullest(clips: list[Clip], buckets: dict[str, list[int]], size: int,
             cfg: "CompilerCfg", done: set[str]) -> tuple[str, list[int]]:
    """The fullest bucket that can fill a short, as (label, indices), or ("", []).

    Every candidate is read back to the model before it goes in: the label came out of
    one shot at temperature 0, and a bucket that cannot fill honestly is skipped for the
    next one down. Labels in `done` are skipped whole: another short of this batch is
    already named after them.
    """
    for label, found in sorted(buckets.items(), key=lambda item: -len(item[1])):
        if len(found) < size or label in done:
            continue
        picked: list[int] = []
        for index in found:
            if fits_tag(clips[index].seen, label, cfg):
                picked.append(index)
                if len(picked) == size:
                    return label, picked
        logger.info(f"'{label}': only {len(picked)} of {len(found)} clips really fit it")
    return "", []


# words that turn a heading into a label for nothing: "TOP 5 WATER SCENES" is a filename
# words that name nothing in the last slot of a heading: "TOP 5 PET FUN" and "TOP 5
# WATCHING ACTS" both came back from the model before this list grew
_MOOD = {"moment", "vibe", "chaos", "compilation", "clip", "animal", "creature", "fail",
         "top", "scene", "thing", "time", "video", "act", "fun", "pet", "antic", "action",
         "adventure", "stuff", "show", "life", "style", "energy", "level", "mode"}

# the per-set ban is built from what vision named in these five clips; the model still
# reaches for a species none of them are ("BLANKET BUNNIES" over five cats)
_SPECIES = {"cat", "kitten", "kitty", "dog", "puppy", "pup", "doggo", "bunny", "rabbit",
            "bird", "parrot", "duck", "goose", "chicken", "hen", "cow", "goat", "sheep",
            "pig", "horse", "monkey", "bear", "panda", "fox", "squirrel", "hamster",
            "guinea", "ferret", "otter", "raccoon", "meerkat", "lizard", "turtle", "fish",
            "frog", "lion", "tiger", "elephant", "penguin", "owl", "deer", "donkey"}


def name_theme(tag: str, clips: list[Clip], size: int, cfg: "CompilerCfg",
               fallback: str = "", done: set[str] | None = None) -> str:
    """The heading for a bucket of `tag` clips, e.g. "stealing food" -> "SNACK BANDITS".

    Two tries, then the tag itself: the model reaches for a mood word or names the animal
    often enough ("CAT WATCHING MOMENTS") that the prompt alone does not hold it, and a
    plain "Watching Something" over the ranking beats a heading that says nothing.

    ``done`` is the batch's headings, extended in place: the same label can name a second
    short late in a batch, but not under the same name.
    """
    taken = set() if done is None else done
    banned = _MOOD.union(_SPECIES).union(
        word.lower().strip(".,").rstrip("s") for clip in clips for word in clip.animal.split()
    )
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": {
            "type": "object", "properties": {"theme": {"type": "string"}}, "required": ["theme"],
        },
        "options": {"temperature": cfg.temperature},
        "messages": [
            {"role": "system", "content": _THEME.format(tag=tag, size=size)},
            {"role": "user", "content": "Clips:\n" + "\n".join(
                f"- {_hide(clip.seen, clip.animal or clip.category)}" for clip in clips
            )},
        ],
    }
    scenes = [clip.seen.lower() for clip in clips]
    shared = {_stem(word) for word in tag.split()}
    for attempt in range(3):
        try:
            body = _chat(payload, cfg)
            answer = json.loads((body.get("message") or {}).get("content") or "{}").get("theme")
        except (PlanError, json.JSONDecodeError) as exc:
            logger.warning(f"naming unavailable ({exc}); the heading is the plain theme")
            break
        # the prefix comes off before the length cap, or "TOP 3 SNACK BANDITS" is "SNACK"
        theme = _row(_category(str(answer or "")))
        # "TOP 5 GRAZE SQUAD" -- the heading counts five of something, so the last word
        # has to be the plural. A collective singular reads as a typo over the ranking.
        if theme and not theme.split()[-1].lower().endswith("s"):
            logger.info(f"heading '{theme}' is not plural; asking again")
            payload["messages"] = payload["messages"] + [
                {"role": "assistant", "content": theme},
                {"role": "user", "content": f'"{theme}" has to be plural -- the heading reads '
                                            f'"TOP {size} {theme.upper()}". Another one.'},
            ]
            continue
        if theme.lower() in taken:
            complaint = f'"{theme}" already heads another short in this batch.'
        # "TOP 5 STANDING ON HIND LEGS" is the label read out loud, not a name for it
        elif {_stem(word) for word in theme.split()} <= shared:
            complaint = f'"{theme}" is the label itself, not a name for the group.'
        elif fake := [word for word in theme.split() if _made_up(word)]:
            # "TOP 5 PROFESSIONAL STARELERS" -- the model invents a word to reach for a
            # plural noun that does not exist, and it goes on the screen looking like a typo
            complaint = f'"{fake[0]}" is not an English word.'
        elif empty := [word for word in theme.split() if _stem(word) in banned]:
            complaint = f'"{empty[0]}" names nothing.'
        elif odd := [word for word in theme.split()
                     if _stem(word) not in shared and _in_one_clip(word, scenes)]:
            # a word of the label is true of all five by construction -- vision words vary
            # too much for "hind" to be in more than one description of five animals
            # standing on their hind legs
            complaint = (f'"{odd[0]}" is in one of the five clips only, so the heading lies '
                         "about the other four.")
        elif heading_holds(theme, clips, cfg):
            taken.add(theme.lower())
            return theme
        else:
            # the cheap checks cannot see an invented detail: "MIRROR LOVERS" over five
            # clips with no mirror is in none of the descriptions, so it looks made up in
            # the good way. The model reads the five and the heading and says.
            complaint = f"\"{theme}\" is not true of all {size} of them."
        logger.info(f"heading '{theme}': {complaint} asking again")
        # told only the rule it broke, the model rewrites; told nothing, it repeats itself
        payload["messages"] = payload["messages"] + [
            {"role": "assistant", "content": theme},
            {"role": "user", "content": f"{complaint} Another one."},
        ]
    heading = fallback or _HEADINGS.get(tag, tag.title())
    # the model gave up on a label this batch already used: "TOP 5 MORE NOSY NEIGHBOURS"
    # is what a second edition is, and it is not the first one's heading twice
    if heading.lower() in taken:
        heading = f"More {heading}"
    taken.add(heading.lower())
    return heading


def _stem(word: str) -> str:
    return word.lower().strip(".,'\"!?").rstrip("s")


def _in_one_clip(word: str, scenes: list[str]) -> bool:
    """Is `word` a prop out of a single clip -- a mirror, a cake, a blanket?

    An invented word ("Bandits", "Icons") is in none of the descriptions and passes; a
    word in every description is what the five share and passes too. Anything in between
    is the model naming five clips after the one it found most vivid.
    """
    stem = _stem(word)
    if len(stem) < 4:  # "the", "in", "on" -- in some descriptions and not others, always
        return False
    found = sum(stem in scene for scene in scenes)
    return 0 < found < len(scenes)


@lru_cache(maxsize=1)
def _dictionary() -> frozenset[str]:
    """The system word list, or empty when the box does not ship one."""
    try:
        words = Path("/usr/share/dict/words").read_text().split()
    except OSError:
        logger.info("no /usr/share/dict/words here; invented words go unchecked")
        return frozenset()
    return frozenset(word.lower() for word in words)


def _made_up(word: str) -> bool:
    """A word no dictionary has: "STARELERS", the model's own plural of "stare".

    The system list is American, so the British spelling of a real word is checked
    against its American twin -- "neighbours" is not in it and is not invented.
    """
    known = _dictionary()
    if not known:
        return False
    stem = _stem(word)
    # the list is from 1934: "snackers" and "selfies" are not in it and are perfectly good
    # meme words, so a word built off a real one with a live suffix counts as real
    forms = [stem, word.lower(), stem.replace("our", "or")]
    forms += [stem[: -len(suffix)] for suffix in ("ers", "er", "ies", "ie", "ing", "ed")
              if stem.endswith(suffix)]
    return len(stem) > 4 and not any(form in known for form in forms)


def make_line(scene: str, row: str, cfg: "CompilerCfg") -> str:
    """The meme caption along the bottom of one clip.

    One call per clip, not one call for all five: asked for the five together the model
    keeps the order of the ranking rows but drifts on these -- a compilation of animals
    standing up came back with "CAT\'S NEW FRIEND" printed under the meerkats.
    """
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": {
            "type": "object", "properties": {"line": {"type": "string"}}, "required": ["line"],
        },
        "options": {"temperature": cfg.temperature},
        "messages": [
            {"role": "system", "content": _LINE},
            {"role": "user", "content": f'Clip: "{scene}"\nAlready printed over it: "{row}"'},
        ],
    }
    for attempt in range(2):
        try:
            body = _chat(payload, cfg)
            answer = json.loads((body.get("message") or {}).get("content") or "{}").get("line")
        except (PlanError, json.JSONDecodeError) as exc:
            logger.warning(f"bottom caption unavailable ({exc}); the clip goes without one")
            return ""
        line = _line(str(answer or ""))
        if line and line.lower() in scene.lower():
            # the model gave up on the joke and read the description back at us. It is not
            # a caption, it is the prompt, and eight words of it land mid-sentence
            complaint = "repeats the description"
        # printing the ranking row twice on one clip reads as a render bug, and the row is
        # already up there: better the clip carries no bottom caption at all
        elif line.strip(".,!?").lower() == row.strip(".,!?").lower():
            complaint = f"repeats the row '{row}'"
        else:
            return line
        logger.info(f"bottom caption {complaint}; asking again")
    return ""


def make_plan(clips: list[Clip], cfg: "CompilerCfg", category: str) -> Plan:
    """Return the compilation plan for `clips`, in order. Raises PlanError on failure."""
    if not clips:
        raise PlanError("no clips to plan")

    prompt = (
        f"Clips:\n{_describe(clips)}\n\n"
        f'These clips were put together as "TOP {len(clips)} {category}"; every caption '
        f"has to fit under that heading.\n\n"
        f"Invent an on-screen title, a first-second hook and exactly {len(clips)} "
        f"captions -- one per clip, in the same order, each a joke about what its own "
        f"clip shows."
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
        category=category,
        title=str(data.get("title") or "Funny Animals").strip(),
        hook=str(data.get("hook") or "").strip(),
        captions=[_row(str(caption)) for caption in data.get("captions") or []],
        lines=[],
    )
    if len(plan.captions) != len(clips):
        raise PlanError(f"model returned {len(plan.captions)} captions for {len(clips)} clips")

    plan = replace(plan, lines=[
        make_line(_hide(clip.seen, clip.animal or clip.category), row, cfg)
        for clip, row in zip(clips, plan.captions)
    ])

    logger.info(f"plan: {plan.category} -- {plan.title}")
    return plan
