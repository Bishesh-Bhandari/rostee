# rostee
RAG chatbot that answers questions from your business documents with file + page citations

## Run locally

1. `copy .env.example .env`, then fill in `GEMINI_API_KEY` and `APP_PASSWORD` (8+ characters).
2. `docker compose up --build`
3. Open http://localhost:8000 and log in with your `APP_PASSWORD`.

## Deploy to Render (free)

1. Push this repo to GitHub.
2. On Render: **New → Blueprint**, then pick this repo. Render reads `render.yaml`.
3. When it asks, enter `GEMINI_API_KEY` and a long random `APP_PASSWORD`.
4. Wait for the build (a few minutes), then open the `.onrender.com` URL it gives you.

Things to know about the free plan:
- **Uploads don't last.** There's no disk, so uploaded PDFs and the index are wiped
  on every restart, redeploy or sleep. Upload again after it wakes up.
- **It sleeps** after 15 minutes idle; the first visit after that takes ~1 minute.
- **512 MB RAM.** rostee uses ~415 MB after an upload and a few questions. Uploads
  are capped at 20 MB (`MAX_UPLOAD_MB`) so one huge PDF can't crash it.
