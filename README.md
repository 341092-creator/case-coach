# MBA Case-Study Coach (Use case #16)

Streamlit app + Gemini. Student loads a case, picks a framework, gets Socratic questions,
answers them, and receives rubric-based feedback (with a non-AI sanity check).

## Run locally
    pip install -r requirements.txt
    export GEMINI_API_KEY=your_key      # free key: aistudio.google.com
    streamlit run app.py

## Deploy for a shareable link (free, ~5 min)
1. Create a GitHub repo and upload: app.py, coach_core.py, requirements.txt
2. Go to share.streamlit.io -> New app -> pick the repo, main file `app.py`
3. Advanced settings -> Secrets, add:   GEMINI_API_KEY = "your_key"
4. Deploy. Share the `*.streamlit.app` URL.
   (Do NOT commit your key to GitHub. If you skip the secret, users can paste their own key in the sidebar.)

## Tests (no API needed)
    python test_core.py && python test_ui.py
