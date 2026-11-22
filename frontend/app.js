// FastAPI serves this page, so it's on the same origin as the API: plain
// paths like "/health" work and CORS never comes into play.
//
// Text is always set with textContent, never innerHTML: answers and sources
// come from PDFs and the LLM, and innerHTML would run any HTML hidden in them.

const statusEl = document.getElementById("status");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file");
const uploadResult = document.getElementById("upload-result");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");

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

// FastAPI sends `detail` as a string for our own 400s, but as a list for
// validation errors (422), so both shapes need handling.
function errorMessage(data) {
  return typeof data.detail === "string" ? data.detail : "The request was rejected.";
}

uploadForm.addEventListener("submit", async (event) => {
  // Without this the browser does an old-style full-page form submit.
  event.preventDefault();

  const form = new FormData();
  // "file" must match the route's parameter name: upload_document(file: UploadFile).
  form.append("file", fileInput.files[0]);

  const button = uploadForm.querySelector("button");
  button.disabled = true;
  uploadResult.textContent = "Uploading and indexing…";
  try {
    const res = await fetch("/documents", { method: "POST", body: form });
    if (redirectIfLoggedOut(res)) return;
    const data = await res.json();
    if (res.ok) {
      uploadResult.textContent = `✅ ${data.file_name}: ${data.chunks} chunks ready to search`;
      checkHealth(); // the chunk count just changed
    } else {
      uploadResult.textContent = `❌ ${errorMessage(data)}`;
    }
  } catch {
    uploadResult.textContent = "❌ Could not reach the server.";
  } finally {
    button.disabled = false;
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const button = askForm.querySelector("button");
  button.disabled = true;
  answerEl.textContent = "Thinking…";
  sourcesEl.replaceChildren(); // clear the previous question's sources
  try {
    const res = await fetch("/ask", {
      method: "POST",
      // JSON this time, not FormData: the route reads an AskRequest body.
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: questionInput.value }),
    });
    if (redirectIfLoggedOut(res)) return;
    const data = await res.json();
    if (!res.ok) {
      answerEl.textContent = `❌ ${errorMessage(data)}`;
      return;
    }
    answerEl.textContent = data.answer;
    // An <ol> numbers items 1, 2, 3…, matching the [1], [2] markers in the answer.
    for (const source of data.sources) {
      const item = document.createElement("li");
      item.textContent = `${source.file_name}, page ${source.page_number}`;
      item.title = source.text; // full chunk text on hover
      sourcesEl.append(item);
    }
  } catch {
    answerEl.textContent = "❌ Could not reach the server.";
  } finally {
    button.disabled = false;
  }
});

checkHealth();
