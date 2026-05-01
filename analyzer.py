import cv2
import base64
import os
from openai import OpenAI

def extract_frames(video_path, max_frames=10):
    """
    Opens a video file and pulls out evenly spaced frames.
    Returns a list of frames encoded as base64 strings (which is
    the format the OpenAI API expects for images).
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError("Could not open video file. Make sure it's a valid mp4 or mov file.")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video loaded: {total_frames} frames, {fps:.1f} fps, {duration:.1f} seconds")

    # Pick evenly spaced frame positions across the whole clip
    if total_frames <= max_frames:
        frame_positions = list(range(total_frames))
    else:
        step = total_frames / max_frames
        frame_positions = [int(i * step) for i in range(max_frames)]

    frames_b64 = []

    for pos in frame_positions:
        video.set(cv2.CAP_PROP_POS_FRAMES, pos)
        success, frame = video.read()

        if not success:
            continue

        # Resize to keep things fast, 640px wide is plenty for analysis
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            frame = cv2.resize(frame, (640, int(height * scale)))

        # Convert frame to JPEG bytes, then to base64 string
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            continue

        b64 = base64.b64encode(buffer).decode('utf-8')
        frames_b64.append(b64)

    video.release()
    print(f"Extracted {len(frames_b64)} frames successfully")
    return frames_b64

FIFA_RULES = """
You are an expert FIFA-certified soccer referee with 20 years of experience.
Your only job is to analyze video frames and return a JSON verdict.

FIFA LAW 12 — FOULS:
A foul occurs when a player:
- Contacts opponent before ball in a tackle
- Trips, pushes, holds, or charges an opponent
- Jumps at or strikes an opponent
- Handles ball deliberately (not goalkeeper)

CARD RULES — be strict:
- No card: minor or careless foul only
- Yellow: reckless challenge, late tackle, tactical foul, diving/simulation
- Red: studs up into opponent, two-footed lunge, violent conduct, 
  excessive force that endangers safety, denying obvious goal (DOGSO)
  
If you see studs raised, a two-footed lunge, or a player endangered — it is RED. Not yellow.

SIMULATION:
If a player falls dramatically with no contact visible — this is a dive. 
Call it a foul against the diving player with a yellow card.

INCONCLUSIVE rules — return Inconclusive if:
- The contact point is not visible in any frame
- Players are too small or too far from camera to judge
- Fewer than 3 frames show the relevant moment

OUTPUT RULES:
- Return ONLY a raw JSON object
- No markdown, no code fences, no text before or after the JSON
- Start your response with { and end with }
"""

def analyze_clip(frames_b64, original_call="Not provided"):
    client = OpenAI()
    import json

    actual_call = original_call if original_call != "Not sure / not provided" else "Unknown"

    # Step 1 — ask the model to analyze the play with no knowledge of the call
    analysis_content = [
        {
            "type": "text",
            "text": """You are reviewing sports officiating footage for a referee training tool.

Analyze these frames from a soccer match and return a technical assessment.

Observe and report:
- Player positions and movement
- Where the ball is relative to player contact
- Body position and balance of both players
- Whether contact occurs before or after ball possession changes
- Any loss of balance or falling by either player

Based on your observations, classify the interaction:
- is_foul: did illegal contact occur under standard rules?
- is_simulation: did a player fall without clear contact?
- severity: none / careless / reckless / excessive_force
- card: None / Yellow / Red
- visible: could you clearly see the contact point?
- description: one technical sentence summarizing what occurred

Return ONLY this JSON, starting with { and ending with }, no other text:
{
  "is_foul": true or false,
  "is_simulation": true or false,
  "severity": "none" or "careless" or "reckless" or "excessive_force",
  "card": "None" or "Yellow" or "Red",
  "visible": true or false,
  "description": "One sentence describing exactly what you saw"
}"""
        }
    ]

    for b64 in frames_b64:
        analysis_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low"
            }
        })

    analysis_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": FIFA_RULES},
            {"role": "user", "content": analysis_content}
        ],
        max_tokens=300,
        temperature=0
    )

    raw_analysis = analysis_response.choices[0].message.content.strip()

    try:
        clean = raw_analysis.strip()
        # Strip any markdown fences
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:]
                if part.strip().startswith("{"):
                    clean = part.strip()
                    break
        # Find the JSON object even if there's text around it
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        clean = clean[start:end]
        analysis = json.loads(clean)
    except Exception:
        # Print raw so you can see what the model actually returned
        print(f"RAW MODEL OUTPUT: {raw_analysis}")
        return {
            "verdict": "Inconclusive",
            "confidence": "Low",
            "reasoning": "Could not parse play analysis. Try a clearer video angle.",
            "rule_cited": "N/A",
            "card_recommendation": "N/A"
        }

    # Step 2 — use the analysis to build the verdict, now bringing in the original call
    if not analysis.get("visible", True):
        verdict = "Inconclusive"
        confidence = "Low"
    elif analysis.get("is_simulation"):
        # Simulation means the ref who called a foul was wrong
        if actual_call == "Foul called":
            verdict = "Bad Call"
        elif actual_call == "No foul called":
            verdict = "Fair Call"
        else:
            verdict = "Bad Call"  # diving should always be penalized
        confidence = "High"
    elif analysis.get("is_foul"):
        if actual_call == "Foul called":
            verdict = "Fair Call"
        elif actual_call == "No foul called":
            verdict = "Bad Call"
        else:
            verdict = "Bad Call"  # foul happened regardless
        confidence = "High" if analysis.get("severity") == "excessive_force" else "Medium"
    else:
        # No foul
        if actual_call == "No foul called":
            verdict = "Fair Call"
        elif actual_call == "Foul called":
            verdict = "Bad Call"
        else:
            verdict = "Fair Call"  # no foul, so no call would have been correct
        confidence = "Medium"

    # Build reasoning from the description
    description = analysis.get("description", "No description available.")
    card = analysis.get("card", "None")

    if analysis.get("is_simulation"):
        rule = "Law 12 — Simulation: a player who attempts to deceive the referee by feigning injury or pretending to have been fouled must be cautioned."
        reasoning = f"{description} This appears to be simulation. A yellow card should be issued to the diving player."
    elif analysis.get("is_foul"):
        severity = analysis.get("severity", "careless")
        rule = f"Law 12 — {'Excessive force: a red card offence that endangers the opponent.' if severity == 'excessive_force' else 'Reckless challenge: a yellow card offence.' if severity == 'reckless' else 'Careless challenge: a direct free kick offence.'}"
        reasoning = f"{description} This constitutes a foul under FIFA Law 12."
    else:
        rule = "Law 12 — No foul: legal challenge where the ball was won cleanly or contact was incidental."
        reasoning = f"{description} No foul criteria met under FIFA Law 12."

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "rule_cited": rule,
        "card_recommendation": card
    }