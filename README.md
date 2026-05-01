# RefCheck AI

AI-powered soccer officiating analysis tool built for GDG BorderHack 2026.

## What it does
Upload a short soccer clip and RefCheck AI will analyze the play using 
GPT-4o vision, compare it against official FIFA Laws of the Game, and 
return a verdict: Fair Call, Bad Call, or Inconclusive.

## How it works
1. Video is uploaded through the Streamlit interface
2. OpenCV extracts 10 evenly spaced frames from the clip
3. Frames are sent to GPT-4o along with FIFA Law 12 rules
4. The model returns a structured JSON verdict with reasoning
5. The verdict is displayed with confidence level and rule cited

## Tech stack
- Python
- Streamlit (UI)
- OpenCV (frame extraction)
- OpenAI GPT-4o (vision analysis)
- FIFA Laws of the Game — Law 12 (rule reasoning)

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Set your OpenAI API key: `$env:OPENAI_API_KEY="your-key"`
4. Run: `python -m streamlit run app.py`

## Limitations
- Analysis quality depends on video angle and resolution
- Short clips (under 30 seconds) work best
- Currently supports soccer only
- API will reject certain videos due to "violence"

## License
MIT