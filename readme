# 📄 ATS Resume Checker

A Streamlit app that scores a resume the way an Applicant Tracking System (ATS) and a recruiter would, then lists prioritised, actionable improvements. Powered by Google's Gemini Flash model.

## Features
- Upload a **PDF or DOCX** resume (up to 5 MB)
- Optional **job description** for a tailored keyword match
- Overall **ATS score (0-100)** computed from five weighted categories:
  keywords & relevance (30%), content & impact (25%), ATS-friendly formatting (20%), structure & completeness (15%), readability (10%)
- Strengths, weaknesses, found / missing keywords
- Prioritised improvements (High / Medium / Low) with example rewrites
- Rule-based quick checks (email, phone, LinkedIn, section headings, word count)
- Download the full report as Markdown

> **Note:** The score is an AI-based estimate, not the output of a real ATS. The app analyses extracted text, so it can't see visual layout (columns, graphics, fonts).

## Project structure
```
app.py             # the Streamlit app
requirements.txt   # Python dependencies
README.md
```

## Run locally
1. Install Python 3.10+.
2. Get a free Gemini API key at https://aistudio.google.com/apikey
3. Install and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Paste your API key in the sidebar, **or** set it beforehand:
   - Environment variable: `GEMINI_API_KEY=your_key`
   - or create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your_key"
     ```

## Configuration
| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | none (required) |
| `GEMINI_MODEL` | secrets / env var / sidebar | `gemini-2.5-flash` |

If Google retires or renames a model, change `GEMINI_MODEL` (or edit the sidebar field) to another Flash model name listed at https://ai.google.dev/gemini-api/docs/models.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit your API key or `secrets.toml`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
5. Click **Deploy**.

## Privacy
Resume text is sent to the Gemini API for analysis. The app itself does not store uploaded files. Don't upload resumes you aren't comfortable sending to a third-party API.

## Troubleshooting
- **"Almost no text could be extracted"** - the file is probably a scanned image. Export a text-based PDF or DOCX.
- **"API key was rejected"** - re-check the key; make sure it's from Google AI Studio.
- **"Rate limit or quota reached"** - wait a minute (free tier limits apply) and retry.
- **"Model name was not found"** - update the model name as described above.
