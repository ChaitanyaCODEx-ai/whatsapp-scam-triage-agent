# WhatsApp Scam Triage & Privacy Advisor (student capstone prototype)

Agentic AI that triages reported WhatsApp messages: classify -> retrieve similar known scams (RAG) -> decide warn / report / escalate -> 3 privacy-settings tips -> final report.
Not affiliated with WhatsApp/Meta. Data is synthetic; policies are illustrative.

## Files
- `app.py` Streamlit UI | `agent.py` tools + orchestrator | `rag.py` ChromaDB + sentence-transformers
- `data/scam_examples.json` 44 scam + 6 legitimate examples | `data/policies_and_tips.txt` policies + privacy tips
- `data/test_messages.json` + `eval.py` held-out evaluation | `docs/` PPT outline and video script

## Run locally
1. `python -m venv .venv` then activate it (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Get a free key at https://aistudio.google.com/app/apikey
4. Set it: macOS/Linux `export GEMINI_API_KEY=...` | PowerShell `$env:GEMINI_API_KEY="..."` (or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`)
5. Smoke tests: `python rag.py` then `python agent.py "send me the 6 digit code"`
6. `streamlit run app.py`
7. Optional accuracy numbers for the slides: `python eval.py`

## Deploy on Streamlit Community Cloud
1. Push this folder to a PUBLIC GitHub repo (do not commit secrets; `.gitignore` covers it)
2. https://share.streamlit.io > Create app > pick repo, branch `main`, main file `app.py`
3. Advanced settings: Python 3.11 or 3.12, Secrets: `GEMINI_API_KEY = "your-key"`
4. Deploy (first boot is slow: it downloads the embedding model). Copy the public URL
5. Open the URL before recording/submitting: free apps go to sleep when idle

## Troubleshooting
- Build too big / out of memory: delete `torch`, `sentence-transformers` and the first line from requirements.txt. rag.py then uses Chroma's built-in ONNX build of the same model automatically.
- Model not found error: set `GEMINI_MODEL` (secret or env var) to a current free-tier model listed in AI Studio.
- 429 errors: free tier rate limit. Wait a minute; the app falls back to rule-based mode instead of crashing.
