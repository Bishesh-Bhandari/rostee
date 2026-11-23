// FastAPI serves this page, so it's on the same origin as the API: plain
// paths like "/health" work and CORS never comes into play.
//
// Text is always set with textContent or text nodes, never innerHTML: answers
// and sources come from PDFs and the LLM, and innerHTML would run any HTML
// hidden in them.

const statusEl = document.getElementById("status");
const messagesEl = document.getElementById("messages");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("question");
const fileInput = document.getElementById("file");
const sendButton = composer.querySelector(".send");

// One request at a time keeps the conversation in order.
let busy = false;

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  fileInput.disabled = value;
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    statusEl.textContent = `Online · ${data.chunks} chunks indexed`;
    statusEl.className = "status ok";
  } catch {
    statusEl.textContent = "Server offline";
    statusEl.className = "status down";
  }
}

// The login cookie can expire while the page is open. The server then
// answers 401, and the right move is the login page, not an error message.
function redirectIfLoggedOut(res) {
  if (res.status === 401) {
    window.location.href = "/login.html";
    return true;
  }
  return false;
}

// A crashed or restarting server answers with an HTML error page, not JSON.
// Without this, res.json() throws and it looks like the network is down.
async function readJson(res) {
  try {
    return await res.json();
  } catch {
    return { detail: `The server had a problem (error ${res.status}). Try again in a minute.` };
  }
}

// FastAPI sends `detail` as a string for our own errors, but as a list for
// validation errors (422), so both shapes need handling.
function errorMessage(data) {
  return typeof data.detail === "string" ? data.detail : "The request was rejected.";
}

// --- Messages ---------------------------------------------------------------

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// role: "user" (right, blue) or "bot" (left, with the R avatar).
// Returns the bubble so the caller can fill it.
function addMessage(role) {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  if (role === "bot") {
    const avatar = document.createElement("span");
    avatar.className = "avatar";
    avatar.textContent = "R";
    avatar.setAttribute("aria-hidden", "true");
    row.append(avatar);
  }
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  row.append(bubble);
  messagesEl.append(row);
  scrollToBottom();
  return bubble;
}

function showTyping(bubble, label) {
  const dots = document.createElement("span");
  dots.className = "typing";
  dots.append(...Array.from({ length: 3 }, () => document.createElement("span")));
  bubble.replaceChildren(dots);
  if (label) {
    const text = document.createElement("span");
    text.className = "typing-label";
    text.textContent = label;
    bubble.append(text);
  }
  scrollToBottom();
}

// Gemini writes light markdown: "* " bullets, **bold**, and [1] markers that
// point at the sources. Each piece becomes a real element, built safely.
function renderText(bubble, text) {
  bubble.replaceChildren();
  let list = null;
  for (const line of text.split("\n")) {
    const bullet = line.match(/^\s*[*-]\s+(.*)$/);
    if (bullet) {
      if (!list) {
        list = document.createElement("ul");
        bubble.append(list);
      }
      const item = document.createElement("li");
      appendInline(item, bullet[1]);
      list.append(item);
    } else if (line.trim()) {
      list = null;
      const paragraph = document.createElement("p");
      appendInline(paragraph, line.trim());
      bubble.append(paragraph);
    } else {
      list = null;
    }
  }
}

function appendInline(parent, text) {
  // The capture group keeps the matched **bold** and [n] parts in the result.
  for (const part of text.split(/(\*\*[^*]+\*\*|\[\d+\])/)) {
    if (!part) continue;
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      parent.append(strong);
    } else if (/^\[\d+\]$/.test(part)) {
      const cite = document.createElement("sup");
      cite.className = "cite";
      cite.textContent = part;
      parent.append(cite);
    } else {
      parent.append(document.createTextNode(part));
    }
  }
}

// Each source is a <details>: click it to open the exact passage the answer
// used. [1] in the answer is the first source, matching the server's order.
function renderSources(bubble, sources) {
  if (!sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  sources.forEach((source, i) => {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `[${i + 1}] ${source.file_name} · page ${source.page_number}`;
    const excerpt = document.createElement("p");
    excerpt.className = "excerpt";
    excerpt.textContent = source.text;
    details.append(summary, excerpt);
    box.append(details);
  });
  bubble.append(box);
}

// --- Upload -----------------------------------------------------------------

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  // Cleared right away so picking the same file again still fires "change".
  fileInput.value = "";
  if (!file || busy) return;

  addMessage("user").textContent = `📄 ${file.name}`;
  const reply = addMessage("bot");
  showTyping(reply, "Reading and indexing your PDF…");
  setBusy(true);
  try {
    const form = new FormData();
    // "file" must match the route's parameter name: upload_document(file: UploadFile).
    form.append("file", file);
    const res = await fetch("/documents", { method: "POST", body: form });
    if (redirectIfLoggedOut(res)) return;
    const data = await readJson(res);
    if (res.ok) {
      const name = document.createElement("strong");
      name.textContent = data.file_name;
      reply.replaceChildren(
        "✅ ",
        name,
        ` is ready: ${data.chunks} chunks indexed. Ask me anything about it.`,
      );
      checkHealth(); // the chunk count just changed
    } else {
      reply.textContent = `❌ ${errorMessage(data)}`;
    }
  } catch {
    reply.textContent = "❌ Could not reach the server.";
  } finally {
    setBusy(false);
    scrollToBottom();
  }
});

// --- Ask --------------------------------------------------------------------

composer.addEventListener("submit", async (event) => {
  // Without this the browser does an old-style full-page form submit.
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || busy) return;

  addMessage("user").textContent = question;
  questionInput.value = "";
  autoGrow();
  const reply = addMessage("bot");
  showTyping(reply);
  setBusy(true);
  try {
    const res = await fetch("/ask", {
      method: "POST",
      // JSON this time, not FormData: the route reads an AskRequest body.
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (redirectIfLoggedOut(res)) return;
    const data = await readJson(res);
    if (!res.ok) {
      reply.textContent = `❌ ${errorMessage(data)}`;
      return;
    }
    renderText(reply, data.answer);
    renderSources(reply, data.sources);
  } catch {
    reply.textContent = "❌ Could not reach the server.";
  } finally {
    setBusy(false);
    scrollToBottom();
    questionInput.focus();
  }
});

// Enter sends, Shift+Enter adds a new line, like most chat apps.
// isComposing: don't send while a keyboard for another script is mid-word.
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

// The box grows with the question, up to a limit, then scrolls.
function autoGrow() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
}
questionInput.addEventListener("input", autoGrow);

renderText(
  addMessage("bot"),
  "Hi, I'm **rostee** 👋\nAttach a PDF with the 📎 button, then ask me anything about it. Every answer tells you the file and page it came from.",
);
checkHealth();
