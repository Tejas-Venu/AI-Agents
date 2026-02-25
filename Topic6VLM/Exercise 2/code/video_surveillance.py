"""
video_surveillance.py

Extract frames every N seconds from a video, optionally pre-screen with a local HOG detector,
ask LLaVA (via Ollama) whether a person is present in positive frames, and write detected
enter/exit times. This corrected version **disables aggressive smoothing** and includes
debug prints so you can see raw and processed detections.

Usage:
    python video_surveillance.py --video "C:\path\to\video.mp4" --interval 2 --use_local_detector

Dependencies:
    pip install opencv-python ollama pillow
    (use quotes around Windows paths that contain spaces)
"""

import argparse
import os
import re
import sys
import threading
from typing import List, Tuple
import ollama
import cv2
from PIL import Image

MODEL = "llava:7b-v1.6-mistral-q4_0"   
FRAME_MAX_SIDE = 800                   
SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "Answer concisely. For the question 'Is there a person in this image?' reply with 'Yes' or 'No' "
    "optionally followed by a short confidence like 'Yes (90%)' or a brief note."
)



def extract_frames(video_path: str, interval_seconds: float) -> Tuple[List[str], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0.0

    interval_frames = max(1, int(round(fps * interval_seconds)))
    actual_interval_seconds = interval_frames / fps

    out_dir = "extracted_frames"
    os.makedirs(out_dir, exist_ok=True)

    frame_paths = []
    frame_num = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval_frames == 0:
            path = os.path.join(out_dir, f"frame_{saved_idx:04d}.jpg")
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(path, format="JPEG", quality=85)
            frame_paths.append(path)
            saved_idx += 1
        frame_num += 1

    cap.release()
    print(f"Extracted {len(frame_paths)} frames every ~{actual_interval_seconds:.2f}s (fps={fps:.2f}, total_frames={total_frames})")
    return frame_paths, actual_interval_seconds


def maybe_resize(image_path: str, max_side: int = FRAME_MAX_SIDE) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) <= max_side:
        return image_path
    scale = max_side / max(w, h)
    out = image_path.rsplit(".", 1)[0] + f"_resized_{max_side}.jpg"
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(out, format="JPEG", quality=85)
    return out


def build_ollama_messages_for_frame(image_path: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Is there a person in this image? Answer 'Yes' or 'No' (optionally a short confidence).", "images": [image_path]},
    ]


def ask_llava_for_frames(frame_paths: List[str]) -> List[str]:
    replies = []
    for i, p in enumerate(frame_paths):
        p2 = maybe_resize(p, FRAME_MAX_SIDE)
        messages = build_ollama_messages_for_frame(p2)
        try:
            resp = ollama.chat(model=MODEL, messages=messages)
            text = resp.get("message", {}).get("content", "") or ""
            text = text.strip()
        except Exception as e:
            text = f"[error contacting ollama: {e}]"
        print(f"[{i:03d}] frame='{p}' -> LLaVA: {text!r}")
        replies.append(text)
    return replies


_yes_no_re = re.compile(r'\b(yes|no)\b', re.IGNORECASE)


def parse_llava_yes_no(reply: str) -> bool:
    if not reply:
        return False
    m = _yes_no_re.search(reply)
    if m:
        return m.group(1).lower() == "yes"
    positive_keywords = ["person", "people", "human", "someone"]
    negative_keywords = ["no person", "empty", "no people", "nobody", "none", "no humans"]
    rep_low = reply.lower()
    for nk in negative_keywords:
        if nk in rep_low:
            return False
    for pk in positive_keywords:
        if pk in rep_low:
            return True
    return False


def local_hog_person_detector(frame_path: str) -> bool:
    img = cv2.imread(frame_path)
    if img is None:
        return False
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if len(rects) > 0:
        if weights is None:
            return True
        try:
            return any(float(w) > 0.4 for w in weights)
        except Exception:
            return True
    return False


def compute_entry_exit_intervals(detections: List[bool], frame_interval_seconds: float) -> List[Tuple[float, float]]:
    intervals = []
    n = len(detections)
    state = False
    enter_idx = None
    for i, present in enumerate(detections):
        if not state and present:
            state = True
            enter_idx = i
        elif state and not present:
            exit_idx = i
            enter_time = enter_idx * frame_interval_seconds
            exit_time = exit_idx * frame_interval_seconds
            intervals.append((enter_time, exit_time))
            state = False
            enter_idx = None
    if state and enter_idx is not None:
        enter_time = enter_idx * frame_interval_seconds
        exit_time = (n - 1) * frame_interval_seconds
        intervals.append((enter_time, exit_time))
    return intervals


def main():
    parser = argparse.ArgumentParser(description="Video -> frames -> ask LLaVA whether a person is present")
    parser.add_argument("--video", required=True, help="Path to input video file (use quotes if path has spaces)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between sampled frames (default 2.0)")
    parser.add_argument("--out", default="person_times.txt", help="Output text file for enter/exit times")
    parser.add_argument("--use_local_detector", action="store_true", help="Use local HOG pre-screen and query LLaVA only for positives")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print("Video file not found:", args.video, file=sys.stderr)
        sys.exit(2)

    print("Extracting frames...")
    frames, actual_interval = extract_frames(args.video, args.interval)
    if not frames:
        print("No frames extracted; exiting.", file=sys.stderr)
        sys.exit(1)

    detections: List[bool] = []
    if args.use_local_detector:
        print("Using local HOG detector to pre-screen frames (only positives will be asked to LLaVA).")
        local_results = [local_hog_person_detector(p) for p in frames]
        to_query_indices = [i for i, val in enumerate(local_results) if val]
        sel_map = {}
        if to_query_indices:
            print(f"Local detector flagged {len(to_query_indices)}/{len(frames)} frames. Querying LLaVA for those...")
            selected_paths = [frames[i] for i in to_query_indices]
            replies_for_selected = ask_llava_for_frames(selected_paths)
            parsed_selected = [parse_llava_yes_no(r) for r in replies_for_selected]
            sel_map = dict(zip(to_query_indices, parsed_selected))
        for i in range(len(frames)):
            if local_results[i]:
                detections.append(sel_map.get(i, False))
            else:
                detections.append(False)
    else:
        replies = ask_llava_for_frames(frames)
        detections = [parse_llava_yes_no(r) for r in replies]


    frame_times = [i * actual_interval for i in range(len(frames))]
    print("\nDEBUG: frame times (s):")
    print(frame_times)
    print("DEBUG: raw detections (per sampled frame):")
    print(detections)
 
    smoothed = detections 

    print("DEBUG: smoothed detections (after processing):")
    print(smoothed)

    intervals = compute_entry_exit_intervals(smoothed, actual_interval)

    with open(args.out, "w") as f:
        if not intervals:
            line = "No person detected in video (according to LLaVA + processing)."
            print(line)
            f.write(line + "\n")
        else:
            print("Detected person entry/exit intervals (seconds):")
            for enter, exit in intervals:
                s = f"ENTER: {enter:.2f}s    EXIT: {exit:.2f}s"
                print(s)
                f.write(s + "\n")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()