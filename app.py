import streamlit as st
import tempfile
import os
import json
from analyzer import extract_frames, analyze_clip

# --- Page config ---
st.set_page_config(
    page_title="RefCheck AI",
    page_icon="⚽",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #0f1117;
        color: #ffffff;
    }
    
    /* Center the title block */
    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    
    .title-block h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    
    .title-block p {
        color: #9ca3af;
        font-size: 1rem;
    }

    /* Primary button */
    .stButton > button {
        background-color: #16a34a;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #15803d;
        color: white;
    }

    /* Verdict cards */
    .verdict-fair {
        background-color: #14532d;
        border-left: 6px solid #16a34a;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    
    .verdict-bad {
        background-color: #450a0a;
        border-left: 6px solid #dc2626;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    
    .verdict-inconclusive {
        background-color: #422006;
        border-left: 6px solid #d97706;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }

    .verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
    }

    .verdict-confidence {
        font-size: 0.9rem;
        color: #9ca3af;
    }

    /* Info boxes */
    .info-box {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border: 1px solid #2d3148;
    }

    .info-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.3rem;
    }

    .info-value {
        font-size: 0.95rem;
        color: #e5e7eb;
    }

    /* Card badges */
    .card-yellow {
        display: inline-block;
        background-color: #854d0e;
        color: #fef08a;
        padding: 0.3rem 0.9rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .card-red {
        display: inline-block;
        background-color: #450a0a;
        color: #fca5a5;
        padding: 0.3rem 0.9rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .card-none {
        display: inline-block;
        background-color: #1e2130;
        color: #9ca3af;
        padding: 0.3rem 0.9rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #4b5563;
        font-size: 0.75rem;
        padding: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="title-block">
    <h1>📝 RefCheck AI </h1>
    <p>Your very own AI VAR — upload a clip and get an instant verdict</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Input section ---
col1, col2 = st.columns(2)

with col1:
    sport = st.selectbox(
        "Sport",
        ["⚽Soccer"],
        help="More sports coming soon.."
    )

with col2:
    original_call = st.selectbox(
        "Original referee call",
        ["Foul called", "No foul called", "Not sure / not provided"]
    )

uploaded_file = st.file_uploader(
    "Upload a short video clip (mp4, mov, avi — max 30 seconds recommended)",
    type=["mp4", "mov", "avi"]
)

analyze_button = st.button("Analyze Clip 📝", type="primary", use_container_width=True)

# --- Analysis ---
if analyze_button:
    if uploaded_file is None:
        st.error("Please upload a video clip first.")
    else:
        # Save uploaded file to a temp location so OpenCV can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Frame extraction
            with st.spinner("Extracting frames from clip..."):
                frames = extract_frames(tmp_path, max_frames=10)
                st.success(f"Extracted {len(frames)} frames from your clip.")

            # AI analysis
            with st.spinner("Analyzing play with our referees..."):
                result = analyze_clip(frames, original_call=original_call)

            st.divider()

            # --- Verdict display ---
            # --- Verdict display ---
            verdict = result.get("verdict", "Inconclusive")
            confidence = result.get("confidence", "Low")
            reasoning = result.get("reasoning", "No reasoning provided.")
            rule_cited = result.get("rule_cited", "N/A")
            card = result.get("card_recommendation", "N/A")

            if verdict == "Fair Call":
                st.markdown(f"""
                <div class="verdict-fair">
                    <div class="verdict-title">Fair Call</div>
                    <div class="verdict-confidence">Confidence: {confidence}</div>
                </div>
                """, unsafe_allow_html=True)
            elif verdict == "Bad Call":
                st.markdown(f"""
                <div class="verdict-bad">
                    <div class="verdict-title">Bad Call</div>
                    <div class="verdict-confidence">Confidence: {confidence}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-inconclusive">
                    <div class="verdict-title">Inconclusive</div>
                    <div class="verdict-confidence">Confidence: {confidence}</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Reasoning
            st.markdown('<div class="info-box"><div class="info-label">Reasoning</div><div class="info-value">' + reasoning + '</div></div>', unsafe_allow_html=True)

            # Rule cited
            st.markdown('<div class="info-box"><div class="info-label">Rule Applied</div><div class="info-value">' + rule_cited + '</div></div>', unsafe_allow_html=True)

            # Card recommendation
            if card == "Yellow":
                st.markdown('<div class="info-box"><div class="info-label">Card Recommendation</div><span class="card-yellow">Yellow Card</span></div>', unsafe_allow_html=True)
            elif card == "Red":
                st.markdown('<div class="info-box"><div class="info-label">Card Recommendation</div><span class="card-red">Red Card</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box"><div class="info-label">Card Recommendation</div><span class="card-none">None</span></div>', unsafe_allow_html=True)

            st.divider()

            # with st.expander("View raw AI response"):
            #     st.json(result)

            # Confidence badge
            st.markdown(f"**Confidence:** {confidence}")

    #        st.divider()

            # # Reasoning
            # st.markdown("### 📋 Reasoning")
            # st.write(reasoning)

            # # Rule cited
            # st.markdown("### 📖 Rule Applied")
            # st.info(rule_cited)

            # # Card recommendation
            # if card == "Yellow":
            #     st.markdown("### 🟨 Card Recommendation: Yellow Card")
            # elif card == "Red":
            #     st.markdown("### 🟥 Card Recommendation: Red Card")
            # elif card == "None":
            #     st.markdown("### Card Recommendation: None")

            st.divider()

            # Raw JSON expander for judges who want to see under the hood
            with st.expander("View raw AI response"):
                st.json(result)

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")

        finally:
            # Clean up the temp file
            os.unlink(tmp_path)

# --- Footer ---
# --- Footer ---
st.markdown(
    '<div class="footer">RefCheck AI · GDG BorderHack 2026 · Powered by GPT-4o · FIFA Laws of the Game</div>',
    unsafe_allow_html=True

)