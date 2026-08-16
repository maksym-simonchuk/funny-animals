/* Popup: server settings, the tag filter, scroll pacing and the run switch.
 *
 * Every control writes straight into chrome.storage.local — the content script
 * watches the same keys, so there is no message plumbing and no `tabs` permission.
 */

const NUMBER_FIELDS = [
  "minDuration",
  "maxDuration",
  "stepPx",
  "minInterval",
  "maxInterval",
  "pauseAfter",
  "pauseSeconds",
  "maxScrolls",
];
const TEXT_FIELDS = ["animalTags", "funnyTags"];

const $ = (id) => document.getElementById(id);

function renderStats(state) {
  const status = state.scrollStatus || {};
  $("matched").textContent = status.matched || 0;
  $("skipped").textContent = status.skipped || 0;
  $("queueSize").textContent = state.queueSize || 0;
  $("status").textContent = status.state
    ? `${status.state} · постов пройдено: ${status.steps || 0}`
    : "Ожидание";
  $("run").textContent = state.running ? "Стоп" : "Старт";
  $("run").classList.toggle("running", Boolean(state.running));
}

async function saveScroll() {
  const scroll = {};
  for (const field of NUMBER_FIELDS) scroll[field] = Number($(field).value);
  for (const field of TEXT_FIELDS) scroll[field] = $(field).value;
  await chrome.storage.local.set({ scroll });
}

async function init() {
  const state = await chrome.storage.local.get([
    "serverUrl",
    "token",
    "scroll",
    "running",
    "queueSize",
    "scrollStatus",
  ]);

  $("serverUrl").value = state.serverUrl || "http://127.0.0.1:8000";
  $("token").value = state.token || "";

  const scroll = { ...SCROLL_DEFAULTS, ...(state.scroll || {}) };
  for (const field of [...NUMBER_FIELDS, ...TEXT_FIELDS]) $(field).value = scroll[field];

  renderStats(state);
}

for (const field of ["serverUrl", "token"]) {
  $(field).addEventListener("change", () =>
    chrome.storage.local.set({ [field]: $(field).value.trim() })
  );
}

for (const field of [...NUMBER_FIELDS, ...TEXT_FIELDS]) {
  $(field).addEventListener("change", saveScroll);
}

$("run").addEventListener("click", async () => {
  const { running } = await chrome.storage.local.get("running");
  if (!running) await saveScroll(); // start from what is currently in the form
  await chrome.storage.local.set({ running: !running });
});

$("flush").addEventListener("click", () => chrome.runtime.sendMessage({ type: "flush" }));

$("clear").addEventListener("click", () =>
  chrome.runtime.sendMessage({ type: "clear" }).then(() => {
    chrome.storage.local.set({ scrollStatus: null });
  })
);

chrome.storage.onChanged.addListener((_changes, area) => {
  if (area !== "local") return;
  chrome.storage.local.get(["running", "queueSize", "scrollStatus"]).then(renderStats);
});

init();
