/* Batches collected URLs and posts them to the local queue server.
 *
 * Nothing is lost when the server is down: the pending batch stays in
 * chrome.storage.local and is retried on the next tick.
 */

const DEFAULTS = {
  serverUrl: "http://127.0.0.1:8000",
  token: "",
  batchSize: 25,
  flushMs: 2000,
};

let pending = [];
let flushTimer = null;
let sessionCount = 0;

/* An unattended window has nobody to open the popup in it, so `app.py browse` leaves
 * the settings in a file next to this one: an extension cannot be handed anything
 * from outside, but it can read its own folder. No file — nothing changes, and the
 * popup stays the only way in.
 */
async function loadSession() {
  let session;
  try {
    const response = await fetch(chrome.runtime.getURL("session.json"));
    session = await response.json();
  } catch (error) {
    return; // an ordinary profile: wait to be started by hand
  }
  const autostart = Boolean(session.autostart);
  await chrome.storage.local.set({
    serverUrl: session.serverUrl || DEFAULTS.serverUrl,
    token: session.token || "",
    // what tells the content script that a hidden tab is not a reason to stop
    unattended: autostart,
    running: autostart,
  });
}

async function loadPending() {
  const stored = await chrome.storage.local.get("pending");
  pending = stored.pending || [];
}

async function savePending() {
  await chrome.storage.local.set({ pending });
}

async function setBadge(text, color) {
  await chrome.action.setBadgeText({ text });
  if (color) await chrome.action.setBadgeBackgroundColor({ color });
}

async function flush() {
  clearTimeout(flushTimer);
  flushTimer = null;
  if (!pending.length) return;

  const settings = { ...DEFAULTS, ...(await chrome.storage.local.get(["serverUrl", "token"])) };
  if (!settings.token) {
    await setBadge("KEY", "#b00020");
    return;
  }

  const batch = pending.slice(0, 100);
  let response;
  try {
    response = await fetch(`${settings.serverUrl.replace(/\/+$/, "")}/queue/urls`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Ingest-Token": settings.token },
      body: JSON.stringify({ urls: batch }),
    });
  } catch (error) {
    await setBadge("OFF", "#b00020"); // server not running — keep the batch for later
    return;
  }

  if (!response.ok) {
    await setBadge(response.status === 401 ? "KEY" : String(response.status), "#b00020");
    return;
  }

  const result = await response.json();

  if (result.accepted === 0 && result.duplicates === 0) {
    // The queue is full: keep the batch, stop the run until it is drained.
    await chrome.storage.local.set({ running: false, queueSize: result.queue_size });
    await setBadge("FULL", "#b00020");
    return;
  }

  pending = pending.slice(batch.length);
  await savePending();
  sessionCount += result.accepted;
  await chrome.storage.local.set({ queueSize: result.queue_size, sessionCount });
  await setBadge(String(sessionCount), "#1a7f37");
}

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type !== "urls") return;
  const fresh = message.urls.filter((url) => !pending.includes(url));
  if (!fresh.length) return;

  pending.push(...fresh);
  savePending();
  if (pending.length >= DEFAULTS.batchSize) flush();
  else if (!flushTimer) flushTimer = setTimeout(flush, DEFAULTS.flushMs);
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "flush") {
    flush().then(() => sendResponse({ ok: true }));
    return true;
  }
  if (message?.type === "clear") {
    pending = [];
    sessionCount = 0;
    savePending().then(() => setBadge("", null)).then(() => sendResponse({ ok: true }));
    return true;
  }
});

loadPending();
loadSession();
