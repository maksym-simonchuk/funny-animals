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
    "These are three frames of one short video clip, in the order they happen. Answer with "
    "two fields. \"animal\" is what the animal is, one or two words, the breed if you can "
    "tell it. \"scene\" is one short factual clause about what the animal DOES across the "
    "three -- what it grabs, where it goes, what changes between the first frame and the "
    "last -- and in that clause you call it \"the animal\", never naming it. Describe the "
    "movement, not the furniture: what is funny about a clip is what happens in it, and "
    "\"the animal is lying on a bed\" is true of a clip nobody would watch. No opinions, "
    "no invented details."
)
_LOOK_SCHEMA = {
    "type": "object",
    "properties": {"animal": {"type": "string"}, "scene": {"type": "string"}},
    "required": ["animal", "scene"],
}

# the three questions worth asking of a clip that is about to cost a fifth of a short.
# Yes/no, not "rate this 1 to 5": asked for a number the model answers 3 or 4 about
# everything, and a score that never varies sorts nothing
_RATE = (
    "These are three frames of one short animal video, in the order they happen. Answer "
    "three yes/no questions about it, going only on what you can see.\n"
    "\"funny\": does something silly or unexpected happen -- the animal is caught at "
    "something it should not be doing, ends up somewhere it does not fit, reacts to "
    "something, or moves in a way that plainly did not go to plan? An animal that just "
    "sits, lies or stands there through all three frames is \"no\", however pretty it is.\n"
    "\"human\": is it doing something a person does -- wearing clothes, standing or "
    "sitting upright, holding a thing in its paws like hands, using furniture, a cup or a "
    "screen?\n"
    "\"cute\": is it a baby, or tiny, or fluffy, or curled up against someone -- and close "
    "enough to the camera that its face carries the shot?"
)
_RATE_SCHEMA = {
    "type": "object",
    "properties": {
        name: {"type": "string", "enum": ["yes", "no"]} for name in ("funny", "human", "cute")
    },
    "required": ["funny", "human", "cute"],
}
# funny counts double: this is a compilation of funny animals, and a cute clip where
# nothing happens is what the ranking rows have nothing to joke about
_WEIGHTS = {"funny": 2, "human": 1, "cute": 1}
_UNRATED = 2  # the middle of 0..4: a clip the model never saw is neither pushed nor buried

# the enum is the decoding grammar: an answer other than these two is impossible
_YES_NO = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
}

# One label per clip, so the list is what a compilation can be about. It was twenty, and
# on a pool of 208 clips twenty labels left 43 of them at "other" while the fullest bucket
# was "watching something" -- 27 clips of an animal looking at a thing, which is the
# dullest short this pipeline can build. The ones added since all name something that
# HAPPENS: the vision pass now reads three frames, so movement is finally visible to it
_TAGS = [
    "stealing food", "eating", "drinking", "sleeping", "wearing clothes", "in water",
    "climbing", "falling over", "yelling", "riding something", "playing with a toy",
    "getting a bath", "begging for food", "standing on hind legs", "chasing", "hiding",
    "watching something", "being held", "escaping", "making a mess",
    "running around wildly", "jumping", "stuck somewhere", "startled",
    "staring at the camera", "cuddling", "carrying something big", "refusing to move",
    "dancing", "sitting like a person", "demanding attention", "yawning or stretching",
    "rolling around", "licking something",
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
    "running around wildly": "Zoomie Athletes", "jumping": "Airborne Acrobats",
    "stuck somewhere": "Trapped Explorers", "startled": "Jump Scare Victims",
    "staring at the camera": "Unblinking Judges", "cuddling": "Cuddle Addicts",
    "carrying something big": "Overloaded Movers", "refusing to move": "Immovable Objects",
    "dancing": "Dance Floor Kings", "sitting like a person": "Tiny Humans",
    "demanding attention": "Attention Tyrants", "yawning or stretching": "Sleepy Stretchers",
    "rolling around": "Rolling Disasters", "licking something": "Serial Lickers",
    "a hat": "Hat Enthusiasts", "a blanket": "Blanket Burritos", "a ball": "Ball Obsessives",
    "a shoe": "Shoe Thieves", "a bag": "Bag Inspectors", "a plant": "Plant Destroyers",
    "a lap": "Lap Occupiers", "a door": "Door Negotiators",
}

# the wider net: several labels one heading can still say honestly. The key is the phrase
# every candidate is asked about (`fits_tag`), so a clip that slipped into a member label
# does not ride into the family on it; the value is the labels and the heading to fall
# back on. Used only after single labels run dry -- "TOP 5 SNACK BANDITS" over five
# thieves beats "TOP 5 FOOD CRIMINALS" over five animals merely near food
_FAMILIES: dict[str, tuple[set[str], str]] = {
    "sneaking or stealing": (
        {"stealing food", "hiding", "escaping", "a shoe", "a bag"}, "Sneaky Operators"),
    "eating or drinking": (
        {"eating", "drinking", "stealing food", "begging for food", "fruit", "junk food",
         "a drink", "licking something"}, "Hungry Legends"),
    # the phrase is what the gate reads, so it stays short: asked whether a clip fits
    # "acting like a person -- dressed up, standing upright, riding or watching a screen",
    # the model said yes to one clip in five and the family never filled
    "acting like a human": (
        {"wearing clothes", "standing on hind legs", "riding something", "clothes",
         "a screen", "a mirror", "sitting like a person", "dancing", "a hat"},
        "Human Impersonators"),
    "in or near water": ({"in water", "getting a bath", "water"}, "Splash Squad"),
    "asleep or lounging": (
        {"sleeping", "a bed", "furniture", "being held", "yawning or stretching",
         "a blanket", "a lap"}, "Professional Loungers"),
    "causing chaos": (
        {"making a mess", "falling over", "chasing", "yelling", "running around wildly",
         "jumping", "rolling around", "a plant"}, "Chaos Machines"),
    "a toy or a box in the frame": (
        {"playing with a toy", "a toy", "a box", "a ball"}, "Playtime Professionals"),
    # the families the new labels bring with them. Each one is still a phrase every
    # candidate is asked about, so a clip cannot ride in on a member label alone
    "cuddling up to someone": (
        {"cuddling", "being held", "a lap", "a baby"}, "Professional Snugglers"),
    "startled or stuck": (
        {"startled", "stuck somewhere", "falling over", "a door"}, "Trouble Magnets"),
    "asking a person for something": (
        {"demanding attention", "begging for food", "refusing to move"}, "Tiny Dictators"),
    "moving fast": (
        {"running around wildly", "jumping", "chasing", "dancing"}, "Speed Demons"),
    "staring at something": (
        {"staring at the camera", "watching something", "a mirror", "a screen"},
        "Professional Observers"),
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
    "a hat", "a blanket", "a ball", "a shoe", "a bag", "a plant", "a lap", "a door",
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
    score: int = 0  # 0..4, `rate_clip`: how much of a reason to watch this one is


@dataclass(frozen=True)
class Plan:
    category: str
    title: str
    hook: str
    captions: list[str]
    lines: list[str]  # the meme caption along the bottom of each clip, one per clip


class PlanError(RuntimeError):
    """The model is unreachable or answered with something unusable."""


_COPY_SYSTEM = (
    "You write the post copy that ships with a finished YouTube Shorts compilation of "
    "funny animal clips. Write in English. You are given what each clip shows and the "
    "heading the compilation carries on screen; everything you write has to be true of "
    "those clips. Never name a species the clips do not show, never promise a clip that "
    "is not in the list. The two platforms want opposite things and must not read alike: "
    "YouTube is searchable, so its title says plainly what the video is; TikTok is a "
    "scroll, so its caption is bait -- open on the one clip worth stopping for, and make "
    "the reader want to see how it ends."
)

# The tags people actually search and follow, spelled the way they are typed there. The
# model chooses from this list instead of inventing one: a coined tag like #PetVids is a
# dead end -- no one browses it -- and camel case is not how anybody types a hashtag.
_HASHTAGS = [
    "shorts", "fyp", "foryou", "viral", "funny", "funnyvideos", "comedy", "lol",
    "animals", "animalvideos", "funnyanimals", "cuteanimals", "animallovers", "wildlife",
    "pets", "petsoftiktok", "cute", "dog", "dogs", "dogsoftiktok", "puppy",
    "cat", "cats", "catsoftiktok", "kitten", "bird", "birds", "horse", "farmanimals",
]

# Every post opens on the tags that carry the reach and only then on the ones about this
# particular video: asked for five the model returns three, and a short whose whole tag
# line is #wildlife is a short the feed has nowhere to put
_YT_ALWAYS = ("funnyanimals", "animals", "animalvideos")
_TT_ALWAYS = ("fyp", "viral", "funnyanimals", "animals")
_TAGS_MAX = 8

# YouTube cuts a title off at 100 characters, and it is the searchable half of the copy
_YT_TITLE_MAX = 100
_TITLE_TAG = "#shorts"
_HASHTAG = re.compile(r"\s*#\w+")


def make_copy(clips: list[Clip], plan: Plan, cfg: "CompilerCfg") -> str:
    """The sidecar text that ships next to the .mp4: a YouTube title and description,
    then a clickbait TikTok caption. Raises PlanError on failure."""
    title_max = _YT_TITLE_MAX - len(_TITLE_TAG) - 1  # the tag is appended, not written
    prompt = (
        f"Clips:\n{_describe(clips)}\n\n"
        f'The compilation is on screen as "TOP {len(clips)} {plan.category}".\n\n'
        f"Write:\n"
        f"1. youtube_title -- under {title_max} characters, says what the video is.\n"
        f"2. youtube_description -- two or three sentences on what is in it.\n"
        f"3. tiktok_caption -- one line. Open on the single funniest clip without "
        f"giving away how it ends. Emoji are fine here and nowhere else.\n"
        f"4. youtube_tags -- five to eight of the listed tags, the ones somebody "
        f"looking for this video would search.\n"
        f"5. tiktok_tags -- three to six of them, the ones this video is scrolled past "
        f"under.\n"
        f"No hashtags inside the title, the description or the caption -- the tags are "
        f"the two lists and nothing else."
    )
    tags = {"type": "array", "items": {"type": "string", "enum": _HASHTAGS}}
    payload = {
        "model": cfg.model,
        "stream": False,
        "think": False,
        "format": {
            "type": "object",
            "properties": {
                "youtube_title": {"type": "string"},
                "youtube_description": {"type": "string"},
                "tiktok_caption": {"type": "string"},
                "youtube_tags": {**tags, "minItems": 5, "maxItems": 8},
                "tiktok_tags": {**tags, "minItems": 3, "maxItems": 6},
            },
            "required": [
                "youtube_title", "youtube_description", "tiktok_caption",
                "youtube_tags", "tiktok_tags",
            ],
        },
        "options": {"temperature": cfg.temperature},
        "messages": [
            {"role": "system", "content": _COPY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(2):
        body = _chat(payload, cfg)
        try:
            data = json.loads((body.get("message") or {}).get("content") or "")
        except json.JSONDecodeError as exc:
            raise PlanError(f"the model did not answer with JSON: {exc}") from exc
        title = _untagged(str(data.get("youtube_title", "")))
        if len(title) <= title_max:
            break
        # one more try, then the truncation: a title cut mid-word is worse than a short one
        logger.info(f"youtube title is {len(title)} characters, asking again")
        if attempt:
            title = _clip_words(title, title_max)

    if not title:
        raise PlanError("the model returned an empty youtube title")
    description = _untagged(data.get("youtube_description", ""))
    tiktok = _untagged(data.get("tiktok_caption", ""))
    # #shorts is what makes YouTube treat the upload as a Short, so it is not the model's
    # to leave out: the title carries it whatever the tag lines say
    return (
        f"YOUTUBE SHORTS\n{title} {_TITLE_TAG}\n\n{description}\n"
        f"{_tag_line(_YT_ALWAYS, data.get('youtube_tags'))}\n\n"
        f"TIKTOK\n{tiktok}\n{_tag_line(_TT_ALWAYS, data.get('tiktok_tags'))}\n"
    )


def _untagged(text: object) -> str:
    """`text` with its whitespace collapsed per line and any hashtags stripped out: the
    tag lines are built from the two lists, so a tag in the prose is only a duplicate."""
    lines = (" ".join(_HASHTAG.sub("", line).split()) for line in str(text).splitlines())
    return "\n".join(line for line in lines if line)


def _tag_line(*groups: object) -> str:
    """One line of hashtags, the groups in the order given, without repeats, without
    anything off the list and without the #shorts the title already carries."""
    flat = [value for group in groups if isinstance(group, (list, tuple)) for value in group]
    seen = dict.fromkeys(str(value).lower().lstrip("#") for value in flat)
    keep = [tag for tag in seen if tag in _HASHTAGS and f"#{tag}" != _TITLE_TAG]
    return " ".join(f"#{tag}" for tag in keep[:_TAGS_MAX])


def _clip_words(text: str, limit: int) -> str:
    """`text` cut to `limit` characters on a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,-")


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


def _images(frames: list[Path]) -> list[str]:
    return [base64.b64encode(frame.read_bytes()).decode("ascii") for frame in frames]


def describe_frames(frames: list[Path], cfg: "CompilerCfg") -> tuple[str, str]:
    """What the local vision model sees across `frames` of one clip, as (animal, scene).

    Three frames, not one. A single still of the middle of a segment cannot tell a dog
    mid-zoomies from a dog standing there, so every description came back as furniture --
    "the animal is lying on a bed with a plaid blanket" -- and a ranking row joking about
    that is a row about nothing. What is funny about a clip is what changes in it, and
    with three frames in order the model can finally say what that is.

    The two fields are kept apart on purpose. The scene leaves the animal unnamed, because
    the text model that writes the ranking opens every row with the species it is given
    ("CHIHUAHUA STRAWBERRY THIEF") however plainly the prompt forbids it. The name still
    has to exist, though: without it the heading comes out as "TOP 5 SWEET THIEVES",
    true of nothing in particular, when it should say which animals are in the set.

    Returns ("", "") when vision is switched off or the model is missing: the plan then
    falls back to the database metadata, which is category and tags and nothing else.
    """
    if not cfg.vision_model or not frames:
        return "", ""

    payload = {
        "model": cfg.vision_model,
        "stream": False,
        "think": False,
        "format": _LOOK_SCHEMA,
        "options": {"temperature": 0.2},
        "messages": [{"role": "user", "content": _LOOK, "images": _images(frames)}],
    }
    try:
        body = _chat(payload, cfg)
        data = json.loads((body.get("message") or {}).get("content") or "{}")
    except (PlanError, json.JSONDecodeError) as exc:
        logger.warning(f"vision unavailable ({exc}); captions fall back to metadata")
        return "", ""

    animal = " ".join(str(data.get("animal") or "").split())[:40]
    return animal, " ".join(str(data.get("scene") or "").split())[:200]


def rate_clip(frames: list[Path], cfg: "CompilerCfg") -> int:
    """How much of a reason to watch this clip is, 0 to 4.

    Nothing used to rate the clips at all: a compilation was the first five clips of the
    fullest bucket in database order, so whether it was funny came down to what the
    scraper happened to download last. The label says what a clip is ABOUT, and that is
    the one thing a heading can be true of -- it says nothing about whether the clip is
    worth eight seconds of a viewer's attention.

    Three yes/no answers, weighted: an unexpected turn counts double, acting like a person
    and being outright cute count once each. Enum-constrained like every other gate here,
    because asked to "rate this out of five" the model answers three or four about
    everything and the ranking it produces sorts nothing.

    Returns `_UNRATED` when vision is off or down -- with no clip rated, the order is the
    one it always was, and a single clip the model failed on lands mid-pack instead of
    being buried under everything it never saw.
    """
    if not cfg.vision_model or not frames:
        return _UNRATED

    payload = {
        "model": cfg.vision_model,
        "stream": False,
        "think": False,
        "format": _RATE_SCHEMA,
        "options": {"temperature": 0.0},  # a verdict, not a joke: same clip, same answer
        "messages": [{"role": "user", "content": _RATE, "images": _images(frames)}],
    }
    try:
        body = _chat(payload, cfg)
        answers = json.loads((body.get("message") or {}).get("content") or "{}")
    except (PlanError, json.JSONDecodeError) as exc:
        logger.warning(f"rating unavailable ({exc}); the clip is neither pushed nor buried")
        return _UNRATED
    return sum(weight for name, weight in _WEIGHTS.items() if answers.get(name) == "yes")


_TEXT = ("These pictures are frames of one video clip. Is there ANY text or logo burned "
         "into ANY of them -- a caption, a title, a subtitle, a numbered list, a username "
         "or @handle, a platform watermark such as TikTok or Instagram, an app or studio "
         "name in a corner? Look in all four corners and along both edges of each, however "
         "small or faint it is. Answer yes or no.")


def has_text(frames: list[Path], cfg: "CompilerCfg") -> bool:
    """Does the clip carry burned-in text or someone else's watermark?

    A reel that came with a "Top 10 Funniest Cats" list burned into the picture puts two
    rankings on the screen at once and reads as broken.

    Every frame of the look goes in, in one question: a caption that fades in halfway
    through the segment is not in the middle frame, and one still is all this gate used
    to see.

    The question used to end with "a small username, logo or channel handle does not
    count", and that sentence is what let a TikTok watermark and a "Dola AI" corner mark
    into a finished short. TikTok suppresses reach on video wearing another platform's
    mark, so now anything burned in counts, however small.
    """
    if not cfg.vision_model or not frames:
        return False

    payload = {
        "model": cfg.vision_model,
        "stream": False,
        "think": False,
        "format": _YES_NO,
        "options": {"temperature": 0.0},
        "messages": [{"role": "user", "content": _TEXT, "images": _images(frames)}],
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

    The best bucket goes first -- the one whose top `size` clips the vision pass rated
    highest, not simply the one with the most clips in it. Across a batch that keeps the
    themes moving: each short empties the bucket it used, and the next has to look
    elsewhere. Inside the short the clips are ordered best first, so the ranking opens on
    the strongest one.

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
        # nothing groupable left -- an honest generic heading beats a themed lie. With no
        # theme to be true to, the only thing left to choose on is which clips are worth
        # watching, so the rating picks all five
        logger.info(f"nothing has {size} clips left to share; taking the best-rated clips")
        best = sorted(range(len(clips)), key=lambda index: -clips[index].score)
        return best[:size], _unused("Funny Animals", done)

    logger.info(f"theme: {tag} -- clips {[index + 1 for index in picked]}")
    done.add(tag)
    return picked, name_theme(tag, [clips[index] for index in picked], size, cfg, done=done)


def _fullest(clips: list[Clip], buckets: dict[str, list[int]], size: int,
             cfg: "CompilerCfg", done: set[str]) -> tuple[str, list[int]]:
    """The best bucket that can fill a short, as (label, indices), or ("", []).

    Best, not fullest. Ranked by size alone, a bucket of twenty-seven clips of an animal
    looking at something beat a bucket of six clips of animals losing a fight with a
    blanket, and the short built off it was five animals sitting still under a heading
    that promised a countdown. A bucket is now worth what its best `size` clips are worth
    (`rate_clip`), and how many clips it holds only breaks the tie.

    Inside the bucket the same order decides who gets in: the highest-rated clips are
    offered first, so a short is the best five the theme has rather than the five the
    scraper happened to download last. The picked indices come back in that order too --
    the best clip opens the short, where a viewer decides in one second whether to stay.

    Every candidate is read back to the model before it goes in: the label came out of
    one shot at temperature 0, and a bucket that cannot fill honestly is skipped for the
    next one down. Labels in `done` are skipped whole: another short of this batch is
    already named after them.
    """
    def worth(found: list[int]) -> tuple[int, int]:
        best = sorted((clips[index].score for index in found), reverse=True)[:size]
        return sum(best), len(found)

    for label, found in sorted(buckets.items(), key=lambda item: worth(item[1]), reverse=True):
        if len(found) < size or label in done:
            continue
        picked: list[int] = []
        # stable, so clips the model rated the same stay in pool order -- the newest first
        for index in sorted(found, key=lambda index: -clips[index].score):
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
    # the model gave up on a label this batch already used, so the fallback is the one
    # that has to say it is a second edition rather than repeat the first's heading
    return _unused(fallback or _HEADINGS.get(tag, tag.title()), taken)


def _unused(heading: str, taken: set[str]) -> str:
    """`heading`, turned into one no other short in the batch is already called.

    "TOP 5 MORE NOSY NEIGHBOURS" is what a second edition of a theme is. A third has no
    phrase for it that is not "More More", so from there the edition is numbered -- the
    generic heading is the last resort of a batch that has run out of themes, and it is
    reached often enough in a long run for two names not to be enough.
    """
    candidate, edition = heading, 2
    while candidate.lower() in taken:
        candidate = f"More {heading}" if edition == 2 else f"{heading} {edition}"
        edition += 1
    taken.add(candidate.lower())
    return candidate


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
