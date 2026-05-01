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
You are analyzing video frames from a soccer match to determine whether a foul occurred.

OFFICIAL FIFA LAWS OF THE GAME — KEY FOUL RULES (Law 12):

A direct free kick is awarded if a player commits any of the following offences:
- Kicks or attempts to kick an opponent
- Trips or attempts to trip an opponent (including sliding tackles that hit the player before or instead of the ball)
- Jumps at an opponent
- Charges an opponent in a careless, reckless, or using excessive force manner
- Strikes or attempts to strike an opponent
- Pushes an opponent
- Makes contact with the opponent before the ball during a tackle
- Holds an opponent
- Impedes an opponent with or without contact
- Handball: deliberately handles the ball (except the goalkeeper in their own area)

SEVERITY DEFINITIONS:
- Careless: no special attention needed, just a foul
- Reckless: player shows disregard for danger — yellow card
- Excessive force: endangers opponent — red card

KEY PRINCIPLES FOR YOUR ANALYSIS:
- If a player wins the ball cleanly first, it is generally NOT a foul even if the opponent falls
- Shoulder-to-shoulder charging is legal if the ball is within playing distance
- Incidental contact that does not affect play is generally not a foul
- Simulation (diving) should be noted if evident
- Consider the angle, momentum, and body position of both players
- If the video frames are too blurry, too far away, or the angle is poor, return Inconclusive

YOUR RESPONSE FORMAT — you must return exactly this JSON structure and nothing else:
{
  "verdict": "Fair Call" or "Bad Call" or "Inconclusive",
  "confidence": "High" or "Medium" or "Low",
  "reasoning": "2-3 sentences explaining what you saw and which rule applies",
  "rule_cited": "The specific FIFA Law 12 rule that applies",
  "card_recommendation": "None" or "Yellow" or "Red" or "N/A"
}
"""

def analyze_clip(frames_b64, original_call="Not provided"):
    """
    Sends extracted frames to GPT-4o along with the FIFA rulebook prompt.
    Returns a dictionary with verdict, confidence, reasoning, and rule cited.
    """
    client = OpenAI()

    actual_call = original_call if original_call != "Not sure / not provided" else "Unknown — do not factor this into your verdict"

    content = [
        {
            "type": "text",
            "text": f"""Analyze these video frames from a soccer match.

Original referee call: {actual_call}

Look at all the frames carefully and determine:
1. What physical interaction or play is happening between the players?
2. Does it match any of the foul criteria in the rules?
3. Was the original referee call correct, incorrect, or is there not enough information?

IMPORTANT: Base your verdict purely on what you observe in the frames.
Do not simply agree with the original call. If the original call is
unknown, analyze the play entirely on its own merits and only return
Fair Call if you can clearly see no foul occurred.

Return only the JSON verdict as instructed. No extra text."""
        }
    ]

    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low"
            }
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": FIFA_RULES},
            {"role": "user", "content": content}
        ],
        max_tokens=500,
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    import json
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        result = json.loads(clean.strip())
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw was: {raw}")
        result = {
            "verdict": "Inconclusive",
            "confidence": "Low",
            "reasoning": "The AI response could not be parsed. Please try again.",
            "rule_cited": "N/A",
            "card_recommendation": "N/A"
        }

    return result
