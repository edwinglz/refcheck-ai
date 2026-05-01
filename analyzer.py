import cv2
import base64
import os

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

        # Resize to keep things fast — 640px wide is plenty for analysis
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

