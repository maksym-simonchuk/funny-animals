/* Walks the feed you have open and queues the reels that look like funny animals.
 *
 * One run does three things: scroll from one post to the next, read each post's
 * hashtags and caption, and queue only the posts that match your tag lists — the
 * rest are scrolled past. Posts whose <video> reports a duration outside the
 * configured window are skipped too, so the downloader never fetches them.
 *
 * The scroller itself is deliberately plain: it lands on the start of the next
 * post, waits a configurable interval, takes a long pause every N posts, and
 * stops at the end of the feed, at a hard cap, or when the tab goes hidden.
 * There is no randomised "look like a human" behaviour here — no synthetic
 * wheel/key/mouse events, no back-scrolling to fake organic movement.
 */

// A hashtag page mixes both link shapes: /reel/ for reels, /p/ for ordinary video
// posts. Both are the same shortcode space and yt-dlp downloads either one.
const POST_RE = /\/(reel|p)\/([A-Za-z0-9_-]{5,})/;
const HASHTAG_RE = /#[\p{L}\p{N}_]+/gu;

let settings = { ...SCROLL_DEFAULTS };
let matchers = { animals: null, funny: null };

const decided = new Set(); // shortcodes already judged, so each post is judged once
const counts = { matched: 0, skipped: 0 };

// --- tag filter -------------------------------------------------------------

function compile(list) {
  const words = String(list || "")
    .split(",")
    .map((word) => word.trim().toLowerCase())
    .filter(Boolean)
    .map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!words.length) return null; // an empty list means "no constraint"
  const alternatives = words.join("|");
  return {
    // A hashtag is one glued word, so a hit anywhere inside it is a real hit.
    tag: new RegExp(alternatives, "iu"),
    // In prose, allow a short inflection but require a word edge after it, so
    // "cats" and "собаки" match while "category" does not.
    text: new RegExp(`(?:^|[^\\p{L}])(?:${alternatives})[\\p{L}]{0,3}(?![\\p{L}])`, "iu"),
  };
}

function cardText(card) {
  const alts = Array.from(card.querySelectorAll("img[alt], video[aria-label]"))
    .map((node) => node.getAttribute("alt") || node.getAttribute("aria-label") || "")
    .join(" ");
  return `${card.innerText || ""} ${alts}`;
}

function matches(matcher, text, tags) {
  if (!matcher) return true;
  return tags.some((tag) => matcher.tag.test(tag)) || matcher.text.test(text);
}

// A hashtag page is itself a tag: on /explore/search/?q=%23funnyanimals the grid
// tiles carry no caption, but the page you opened already says what they are.
function pageText() {
  return `${decodeURIComponent(location.href)} ${document.title}`;
}

/** true — queue it, false — scroll past, null — nothing rendered yet, ask again. */
function classify(card, context) {
  const video = card.querySelector("video");
  const duration = video && Number.isFinite(video.duration) ? video.duration : null;
  if (duration !== null && (duration < settings.minDuration || duration > settings.maxDuration)) {
    return false;
  }
  if (!matchers.animals && !matchers.funny) return true; // both lists empty: take everything

  const own = cardText(card);
  const text = `${own} ${context}`;
  const tags = text.match(HASHTAG_RE) || [];
  if (matches(matchers.animals, text, tags) && matches(matchers.funny, text, tags)) return true;
  // A miss is only final once the post itself has rendered its caption.
  return own.trim() ? false : null;
}

// --- posts on the page ------------------------------------------------------

// Instagram's markup is unstable, so find the post box by size: in a grid the link
// is the tile itself, in the feed it is a bare link inside a much larger post.
function cardFor(anchor) {
  const article = anchor.closest("article");
  if (article) return article;
  let element = anchor;
  if (element.getBoundingClientRect().height >= 120) return element; // a grid tile
  while (
    element.parentElement &&
    element.getBoundingClientRect().height < window.innerHeight * 0.4
  ) {
    element = element.parentElement;
  }
  return element;
}

function postCards() {
  const cards = new Map();
  for (const anchor of document.querySelectorAll('a[href*="/reel/"], a[href*="/p/"]')) {
    const match = POST_RE.exec(anchor.getAttribute("href") || "");
    if (!match || cards.has(match[2])) continue;
    cards.set(match[2], { card: cardFor(anchor), url: canonical(match[1], match[2]) });
  }
  return cards;
}

function canonical(kind, shortcode) {
  return `https://www.instagram.com/${kind}/${shortcode}/`;
}

function queue(url) {
  counts.matched += 1;
  chrome.runtime.sendMessage({ type: "urls", urls: [url] }).catch(() => {});
}

function scan() {
  const cards = postCards();
  const context = pageText();
  const fresh = [];
  for (const [shortcode, { card, url }] of cards) {
    if (decided.has(shortcode)) continue;
    const verdict = classify(card, context);
    if (verdict === null) continue; // caption still loading — judge it next round
    decided.add(shortcode);
    if (verdict) fresh.push(url);
    else counts.skipped += 1;
  }
  if (fresh.length) {
    counts.matched += fresh.length;
    chrome.runtime.sendMessage({ type: "urls", urls: fresh }).catch(() => {});
  }
  return cards;
}

// --- one open post ----------------------------------------------------------

const SINGLE_POST_RE = /^\/(reel|p)\/([A-Za-z0-9_-]{5,})/;
const NEXT_LABEL_RE = /next|далее|вперёд|вперед|далі|наступн/i;

/** The post the address bar is on, or null when this is a feed/grid page. */
function currentPost() {
  const match = SINGLE_POST_RE.exec(location.pathname);
  if (!match) return null;
  return {
    shortcode: match[2],
    url: canonical(match[1], match[2]),
    card: document.querySelector("article") || document.body,
  };
}

// The viewer's own arrow. A carousel has its own arrow inside <article>, so skip
// anything nested there — that one only switches slides of the same post.
function nextPostButton() {
  for (const node of document.querySelectorAll("[aria-label]")) {
    if (!NEXT_LABEL_RE.test(node.getAttribute("aria-label") || "")) continue;
    const button = node.closest('button, div[role="button"], a');
    if (button && !button.closest("article")) return button;
  }
  return null;
}

const openPost = { shortcode: null, waits: 0 };

/** false while the caption is still rendering — judge it on the next tick. */
function judgePost(current) {
  if (current.shortcode !== openPost.shortcode) {
    openPost.shortcode = current.shortcode;
    openPost.waits = 0;
  }
  if (decided.has(current.shortcode)) return true;

  const verdict = classify(current.card, pageText());
  if (verdict === null && openPost.waits < 2) {
    openPost.waits += 1;
    return false;
  }
  decided.add(current.shortcode);
  if (verdict) queue(current.url);
  else counts.skipped += 1;
  return true;
}

// Instagram does not always scroll the window: on the explore grid the feed lives
// in a nested column, and scrolling the window there does nothing at all. Ask a
// card which of its ancestors actually moves.
function scrollBox(card) {
  const page = document.scrollingElement || document.documentElement;
  for (let el = card?.parentElement; el && el !== document.body; el = el.parentElement) {
    const overflow = getComputedStyle(el).overflowY;
    if ((overflow === "auto" || overflow === "scroll") && el.scrollHeight > el.clientHeight + 8) {
      return el;
    }
  }
  return page;
}

// Where the box's own top edge sits in the viewport; the page box starts at 0.
function boxTop(box) {
  return box === (document.scrollingElement || document.documentElement)
    ? 0
    : box.getBoundingClientRect().top;
}

// The next post is the closest one whose top edge is still below the box top.
function nextCardOffset(cards, top) {
  let best = null;
  for (const { card } of cards.values()) {
    const offset = card.getBoundingClientRect().top - top;
    if (offset > 8 && (best === null || offset < best)) best = offset;
  }
  return best;
}

// --- scroller ---------------------------------------------------------------

const scroller = {
  running: false,
  steps: 0,
  idleRounds: 0,
  lastHeight: 0,
  timer: null,

  start() {
    if (this.running) return;
    this.running = true;
    this.steps = 0;
    this.idleRounds = 0;
    this.lastPost = null;
    this.lastHeight = document.documentElement.scrollHeight;
    this.report("running");
    this.schedule(0);
  },

  stop(reason) {
    if (!this.running) return;
    this.running = false;
    clearTimeout(this.timer);
    this.timer = null;
    this.report(reason);
    chrome.storage.local.set({ running: false });
  },

  report(state) {
    chrome.storage.local.set({
      scrollStatus: { state, steps: this.steps, ...counts },
    });
  },

  schedule(delay) {
    this.timer = setTimeout(() => this.tick(), delay);
  },

  tick() {
    if (!this.running) return;

    // Never move a tab the user is not looking at.
    if (document.hidden) return this.stop("stopped: tab hidden");
    if (this.steps >= settings.maxScrolls) return this.stop("stopped: reached max scrolls");

    // An open post is paged with the viewer's arrow; a feed or grid is scrolled.
    const current = currentPost();
    const stepped = current ? this.pagePost(current) : this.scrollFeed();
    if (stepped) {
      this.steps += 1;
      this.report("running");
    }
    if (!this.running) return; // the branch above reached the end and stopped

    const { minInterval, maxInterval, pauseAfter, pauseSeconds } = settings;
    // Jitter so lazy-loaded posts have time to settle and requests stay spread out.
    const interval = minInterval + Math.random() * Math.max(0, maxInterval - minInterval);
    const takeLongPause = stepped && pauseAfter > 0 && this.steps % pauseAfter === 0;
    this.schedule(takeLongPause ? pauseSeconds * 1000 : interval);
  },

  scrollFeed() {
    const cards = scan(); // judge what is on screen before moving on

    const box = scrollBox(cards.values().next().value?.card);
    const height = box.scrollHeight;
    const atBottom = box.clientHeight + box.scrollTop >= height - settings.stepPx;
    if (atBottom && height <= this.lastHeight) {
      this.idleRounds += 1;
      if (this.idleRounds >= IDLE_ROUNDS_TO_STOP) {
        this.stop("stopped: end of feed");
        return false;
      }
    } else {
      this.idleRounds = 0;
    }
    this.lastHeight = height;

    // Land on the start of the next post; fall back to a plain step when nothing
    // below is loaded yet, which is what makes the feed load more.
    const offset = nextCardOffset(cards, boxTop(box));
    box.scrollBy(0, offset === null ? settings.stepPx : offset);
    return true;
  },

  // Judge the post that is open, then press "next". One plain click, on the same
  // pacing as the feed — no synthetic input beyond it.
  pagePost(current) {
    if (current.shortcode === this.lastPost) {
      this.idleRounds += 1;
      if (this.idleRounds >= IDLE_ROUNDS_TO_STOP) {
        this.stop("stopped: end of feed"); // the arrow stopped moving us anywhere
        return false;
      }
    } else {
      this.idleRounds = 0;
      this.lastPost = current.shortcode;
    }

    if (!judgePost(current)) return false; // caption has not rendered yet

    const next = nextPostButton();
    if (!next) {
      this.stop("stopped: no next arrow");
      return false;
    }
    next.click();
    return true;
  },
};

document.addEventListener("visibilitychange", () => {
  if (document.hidden) scroller.stop("stopped: tab hidden");
});

// --- wiring -----------------------------------------------------------------

function applyState(state) {
  settings = { ...SCROLL_DEFAULTS, ...(state.scroll || {}) };
  matchers = { animals: compile(settings.animalTags), funny: compile(settings.funnyTags) };
  if (state.running) scroller.start();
  else scroller.stop("idle");
}

const STATE_KEYS = ["running", "scroll"];

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  if (!STATE_KEYS.some((key) => key in changes)) return;
  chrome.storage.local.get(STATE_KEYS).then(applyState);
});

// A page load always stops the run: it only ever starts because you just started it.
chrome.storage.local
  .set({ running: false })
  .then(() => chrome.storage.local.get(STATE_KEYS))
  .then(applyState);
