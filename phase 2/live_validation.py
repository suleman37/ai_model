# ==========================================================
# PHASE 2 — Standalone Live Validation (TFLite + YOLO)
# ==========================================================

import cv2
import numpy as np
import os
import time
import argparse
import requests
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PT = os.path.join(ROOT_DIR, "best.pt") 
MODEL_TFLITE = os.path.join(ROOT_DIR, "models", "best_float16.tflite")

IMAGE_SIZE = 256
PIXELS_PER_CM = 100
CONF_THRESHOLD = 0.25

# ==================== CORE LOGIC ====================

def detect_blue_markers_live(image):
    """Detect physical blue marker dots on a live/normalized ear image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 100, 70])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20: continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        if (4 * np.pi * area / (perimeter * perimeter)) < 0.2: continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            pts.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    pts.sort(key=lambda p: p[1])
    return pts

def get_point_guidance(digital, live):
    dx = digital[0] - live[0]
    dy = digital[1] - live[1]
    dist_px = float(np.hypot(dx, dy))
    tol = 5
    mm_pp = 10.0 / PIXELS_PER_CM

    if dist_px <= tol:
        return "CORRECT ✓", round(dist_px / PIXELS_PER_CM, 3)

    hdir, vdir = "", ""
    if dx > tol: hdir = f"RIGHT {abs(dx)*mm_pp:.1f}mm"
    elif dx < -tol: hdir = f"LEFT {abs(dx)*mm_pp:.1f}mm"
    if dy > tol: vdir = f"DOWN {abs(dy)*mm_pp:.1f}mm"
    elif dy < -tol: vdir = f"UP {abs(dy)*mm_pp:.1f}mm"

    msg = " & ".join(filter(None, [hdir, vdir]))
    return f"Move -> {msg}", round(dist_px / PIXELS_PER_CM, 3)

def segment_and_normalize_v2(image_array, seg_model):
    """YOLO-segment the ear and normalize."""
    results = seg_model.predict(source=image_array, conf=CONF_THRESHOLD, verbose=False)
    r = results[0]

    if r.masks is None or len(r.masks.data) == 0:
        return None

    mask = r.masks.data[0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_NEAREST)

    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0: return None

    cy, cx = coords.mean(axis=0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    dist_y = max(abs(y_max - cy), abs(cy - y_min))
    dist_x = max(abs(x_max - cx), abs(cx - x_min))
    max_radius = max(dist_y, dist_x)
    max_dim = int(max_radius * 2 * 1.15)
    half_dim = max_dim // 2

    x1, y1 = int(cx - half_dim), int(cy - half_dim)
    x2, y2 = x1 + max_dim, y1 + max_dim

    img_h, img_w = image_array.shape[:2]
    safe_x1, safe_y1 = max(0, x1), max(0, y1)
    safe_x2, safe_y2 = min(img_w, x2), min(img_h, y2)
    cropped = image_array[safe_y1:safe_y2, safe_x1:safe_x2]

    pad_top, pad_bottom = max(0, -y1), max(0, y2 - img_h)
    pad_left, pad_right = max(0, -x1), max(0, x2 - img_w)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

    return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

# ==================== MAIN EXECUTION ====================

def main():
    print("Initializing Phase 2 Live Validator...")
    try:
        detect_model = YOLO(MODEL_TFLITE, task='detect')
        seg_model = YOLO(MODEL_PT)
        print("✓ Models Loaded!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=str, default=None)
    parser.add_argument("--side", type=str, default="left")
    args = parser.parse_args()

    # Default points (fallback)
    digital_pts = [(128, 128), (128, 180), (128, 230)] 

    # Try to fetch session points if provided
    if args.session_id:
        try:
            print(f"Fetching points for session {args.session_id} ({args.side})...")
            # Phase 2 runs on 8080
            api_url = f"http://localhost:8080/session-points/{args.session_id}?side={args.side}"
            resp = requests.get(api_url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") and data.get("points"):
                    # Handle both list and dict formats
                    digital_pts = [(p[0], p[1]) if isinstance(p, list) else (p['x'], p['y']) for p in data["points"]]
                    print(f"✓ Successfully loaded {len(digital_pts)} custom points.")
                else:
                    print("⚠ Session has no points. Using defaults.")
            else:
                print(f"⚠ Failed to fetch points (Status {resp.status_code}). Using defaults.")
        except Exception as e:
            print(f"⚠ Error fetching points: {e}. Using defaults.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Optimization state
    frame_count = 0
    last_has_ear = False
    last_norm_ear = None
    last_annotated = None
    last_box = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # Only run detection every 3 frames for smoothness
        if frame_count % 3 == 0:
            detect_results = detect_model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
            last_has_ear = len(detect_results[0].boxes) > 0
            if last_has_ear:
                last_box = detect_results[0].boxes[0].xyxy[0].cpu().numpy()
            else:
                last_box = None

        display_frame = frame.copy()
        
        if last_has_ear:
            # Only run heavy segmentation every 6 frames
            if frame_count % 6 == 0:
                norm_ear = segment_and_normalize_v2(frame, seg_model)
                if norm_ear is not None:
                    last_norm_ear = norm_ear
                    live_pts = detect_blue_markers_live(norm_ear)
                    annotated = norm_ear.copy()
                    used = set()
                    for i, dp in enumerate(digital_pts):
                        cv2.circle(annotated, (int(dp[0]), int(dp[1])), 7, (0, 0, 0), 2)
                        best_d, best_j = 9999, -1
                        for j, lp in enumerate(live_pts):
                            if j in used: continue
                            d = np.hypot(lp[0] - dp[0], lp[1] - dp[1])
                            if d < best_d: best_d, best_j = d, j

                        if best_j != -1 and best_d < IMAGE_SIZE * 0.4:
                            used.add(best_j)
                            lp = live_pts[best_j]
                            msg, err_cm = get_point_guidance(dp, lp)
                            correct = "CORRECT" in msg
                            color = (0, 220, 80) if correct else (0, 165, 255)
                            cv2.circle(annotated, (int(lp[0]), int(lp[1])), 6, (255, 50, 50), -1)
                            cv2.arrowedLine(annotated, (int(lp[0]), int(lp[1])), (int(dp[0]), int(dp[1])), (0, 255, 0), 1)
                            cv2.putText(annotated, msg, (10, 25 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    last_annotated = annotated

            if last_annotated is not None:
                cv2.imshow("Normalized Ear (Phase 2)", last_annotated)
            
            if last_box is not None:
                cv2.rectangle(display_frame, (int(last_box[0]), int(last_box[1])), (int(last_box[2]), int(last_box[3])), (255, 0, 0), 2)
        else:
            cv2.putText(display_frame, "Searching for ear...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Phase 2 Live View", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif ord('1') <= key <= ord('5'):
            num = int(chr(key))
            digital_pts = [(128, 100 + i*40) for i in range(num)]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
