# ==========================================================
# FASTAPI MODULE 2 - Ear Landmark Mapping API
# ==========================================================
# Converts the Colab notebook workflow into a REST API:
#   1. Upload right + left ear images → segment & normalize
#   2. User clicks landmark points on the right ear
#   3. Points are mirrored to the left ear
#   4. Distances between consecutive landmarks are calculated
# ==========================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, List
import numpy as np
import cv2
from PIL import Image
import io
import base64
import tempfile
import os
import uuid
from ultralytics import YOLO
import logging
from blue_point_detector import detect_blue_points, count_blue_points, draw_detected_points
import subprocess
import threading


def resolve_existing_path(*candidates: str) -> str:
    """Return the first existing path, or the first candidate as fallback for logging."""
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = resolve_existing_path(
    os.path.join(BASE_DIR, "best.pt"),
    os.path.join(os.path.dirname(BASE_DIR), "best.pt"),
)
TFLITE_MODEL_PATH = resolve_existing_path(
    os.path.join(BASE_DIR, "best_float16.tflite"),
    os.path.join(BASE_DIR, "models", "best_float16.tflite"),
    os.path.join(os.path.dirname(BASE_DIR), "models", "best_float16.tflite"),
)
PIXELS_PER_CM = 100
IMAGE_SIZE = 256
CONFIDENCE_THRESHOLD = 0.25
MASK_THRESHOLD = 0.5

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Ear Landmark Mapping API",
    description="API for ear segmentation, landmark annotation, mirroring, and distance measurement",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LOAD MODEL ====================
model = None
detect_model = None


def load_model_on_startup():
    global model, detect_model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info(f"✓ YOLO model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"⚠ YOLO model not found at {MODEL_PATH}")

        if os.path.exists(TFLITE_MODEL_PATH):
            detect_model = YOLO(TFLITE_MODEL_PATH, task='detect')
            logger.info(f"✓ TFLite Detection model loaded from {TFLITE_MODEL_PATH}")
        else:
            logger.warning(f"⚠ TFLite model not found at {TFLITE_MODEL_PATH}")
    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}")


load_model_on_startup()


# ==================== PYDANTIC MODELS ====================
class Point(BaseModel):
    x: float
    y: float


class LandmarkRequest(BaseModel):
    points: List[Point]


class MirrorAndMeasureRequest(BaseModel):
    right_ear_points: List[Point]
    session_id: str


# ==================== UTILITY FUNCTIONS ====================


def _round_float(value: float, ndigits: int) -> float:
    """Round a float to ndigits decimal places. Avoids Pyre2 round() false positives."""
    multiplier = 10 ** ndigits
    return float(int(value * multiplier + 0.5)) / multiplier


def image_to_base64(image_array):
    """Convert numpy array (BGR or BGRA) to base64 PNG string"""
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 3:
            # BGR to RGB
            color_converted = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif image_array.shape[2] == 4:
            # BGRA to RGBA for transparency
            color_converted = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGBA)
        else:
            color_converted = image_array
    else:
        color_converted = image_array
        
    pil_image = Image.fromarray(color_converted)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def center_ear(image, mask):
    """Center the ear in the image using the mask centroid"""
    h, w = mask.shape

    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return image

    cy, cx = coords.mean(axis=0)

    shift_x = int(w / 2 - cx)
    shift_y = int(h / 2 - cy)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    centered = cv2.warpAffine(image, M, (w, h))
    return centered


def segment_and_normalize(image_array):
    """
    Segment the ear from an image array and normalize it.
    Returns the normalized ear image (256x256) as a numpy array.
    """
    if model is None:
        raise Exception("Model not loaded")

    # Save to temp file for YOLO prediction
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image_array)

        results = model.predict(source=tmp_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        r = results[0]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if r.masks is None or len(r.masks.data) == 0:
        raise ValueError("No ear detected in the image")

    mask = r.masks.data[0].cpu().numpy()
    mask = (mask > MASK_THRESHOLD).astype(np.uint8)
    mask = cv2.resize(
        mask,
        (image_array.shape[1], image_array.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Use the original image pixels for a natural background instead of a black/transparent mask
    segmented = image_array.copy()
    
    # Calculate crop area centered on the ear's center of mass
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return cv2.resize(segmented, (IMAGE_SIZE, IMAGE_SIZE))

    cy, cx = coords.mean(axis=0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Square calculation: take the largest radius and double it
    dist_y = max(abs(y_max - cy), abs(cy - y_min))
    dist_x = max(abs(x_max - cx), abs(cx - x_min))
    
    # We want a 1:1 square crop. We take the larger dimension for both attributes.
    max_radius = max(dist_y, dist_x)
    max_dim = int(max_radius * 2 * 1.15) # 15% padding
    
    half_dim = max_dim // 2
    x1, y1 = int(cx - half_dim), int(cy - half_dim)
    x2, y2 = x1 + max_dim, y1 + max_dim

    img_h, img_w = segmented.shape[:2]
    
    # Get the portion of the crop box that is inside the original image bounds
    safe_x1, safe_y1 = max(0, x1), max(0, y1)
    safe_x2, safe_y2 = min(img_w, x2), min(img_h, y2)
    
    cropped_square = segmented[safe_y1:safe_y2, safe_x1:safe_x2]
    
    # Pad to restore the full max_dim x max_dim square
    # Using BORDER_REPLICATE to keep the background natural instead of black
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - img_h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - img_w)
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped_square = cv2.copyMakeBorder(cropped_square, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

    # SECURE ASPECT RATIO: Final check to ensure it is perfectly square before resizing
    ch, cw = cropped_square.shape[:2]
    if ch != cw:
        dim = max(ch, cw)
        cropped_square = cv2.copyMakeBorder(
            cropped_square, 0, dim - ch, 0, dim - cw, cv2.BORDER_REPLICATE
        )

    # Normalize to IMAGE_SIZE x IMAGE_SIZE
    # Resizing a square image to a square (256x256) ensures no distortion/compression
    normalized = cv2.resize(
        cropped_square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    
    return normalized


def draw_landmarks(image, points, color=(0, 0, 255), text_color=(0, 255, 255)):
    """Draw numbered landmark points on an image"""
    img = image.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 5, color, -1)
        cv2.putText(
            img,
            str(i + 1),
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )
    return img


def draw_landmarks_with_lines(image, points, color=(0, 0, 255), text_color=(0, 255, 255), line_color=(0, 255, 0)):
    """Draw numbered landmark points with connecting lines on an image"""
    img = image.copy()

    # Draw lines between consecutive points
    for i in range(len(points) - 1):
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
        cv2.line(img, pt1, pt2, line_color, 2)

    # Draw points on top
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 5, color, -1)
        cv2.putText(
            img,
            str(i + 1),
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )
    return img




# ==================== IN-MEMORY SESSION STORE ====================
# Stores normalized ear images per session for the two-step workflow
sessions: Dict[str, Any] = {}


# ==================== API ENDPOINTS ====================


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ear Landmark Mapping API (Module 2)",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "phase_1": {
                "segment": "/segment (POST) - Upload right & left ear images",
                "mirror_and_measure": "/mirror-and-measure (POST) - Send right ear landmarks"
            },
            "phase_1_validation": {
                "validate_frame": "/validate-frame (POST) - Real-time webcam validation with blue markers"
            },
            "phase_2_prepiercing": {
                "detect_prepiercing": "/detect-prepiercing (POST) - Analyze image with blue points marked (PRE-PIERCING BASELINE)",
                "get_prepiercing_baseline": "/get-prepiercing-baseline (POST) - Retrieve saved pre-piercing baseline"
            },
            "session_management": {
                "get_session": "/session/{session_id} (GET) - Get session status",
                "delete_session": "/session/{session_id} (DELETE) - Delete session"
            }
        },
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "image_size": IMAGE_SIZE,
        "pixels_per_cm": PIXELS_PER_CM,
    }


@app.post("/segment")
async def segment_ears(
    rightEar: UploadFile = File(...),
    leftEar: UploadFile = File(...),
):
    """
    Step 1: Upload right and left ear images.
    Both ears are segmented, normalized to 256x256, and centered.
    Returns:
      - session_id: unique ID for this session
      - right_ear_image: base64 PNG of the normalized right ear (for landmark clicking)
      - left_ear_image: base64 PNG of the normalized left ear (preview)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Read right ear
        right_contents = await rightEar.read()
        right_nparr = np.frombuffer(right_contents, np.uint8)
        right_image = cv2.imdecode(right_nparr, cv2.IMREAD_COLOR)
        if right_image is None:
            raise ValueError("Invalid right ear image file")

        # Read left ear
        left_contents = await leftEar.read()
        left_nparr = np.frombuffer(left_contents, np.uint8)
        left_image = cv2.imdecode(left_nparr, cv2.IMREAD_COLOR)
        if left_image is None:
            raise ValueError("Invalid left ear image file")

        # Segment and normalize both
        right_normalized = segment_and_normalize(right_image)
        left_normalized = segment_and_normalize(left_image)

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Store in memory
        sessions[session_id] = {
            "right_ear": right_normalized,
            "left_ear": left_normalized,
        }

        logger.info(f"Session {session_id}: Both ears segmented and stored")

        return JSONResponse(
            {
                "success": True,
                "data": {
                    "session_id": session_id,
                    "image_size": IMAGE_SIZE,
                    "right_ear_image": image_to_base64(right_normalized),
                    "left_ear_image": image_to_base64(left_normalized),
                },
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mirror-and-measure")
async def mirror_and_measure(request: MirrorAndMeasureRequest):
    """
    Step 2: Send the landmarks clicked on the right ear image.
    The API will:
      1. Mirror the points horizontally for the left ear
      2. Draw landmarks on both ear images
      3. Calculate distances between consecutive landmark pairs
    
    Returns:
      - right_ear_landmarks_image: right ear with drawn landmarks
      - left_ear_landmarks_image: left ear with mirrored landmarks
      - right_ear_points: original points
      - left_ear_points: mirrored points
      - distances: list of distances between consecutive points (pixels & cm)
    """
    session_id = request.session_id

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload images first via /segment",
        )

    session = sessions[session_id]
    right_ear = session["right_ear"]
    left_ear = session["left_ear"]

    # Convert points
    right_points = [(p.x, p.y) for p in request.right_ear_points]

    if len(right_points) < 1:
        raise HTTPException(
            status_code=400,
            detail="At least 1 landmark point is required for mirroring",
        )

    # Mirror points: simple horizontal flip (256 - x), y stays the same
    left_points = []
    for x, y in right_points:
        mirrored_x = IMAGE_SIZE - x
        mirrored_y = y

        left_points.append((_round_float(float(mirrored_x), 1), _round_float(float(mirrored_y), 1)))

    # Draw landmarks on right ear
    right_ear_with_landmarks = draw_landmarks_with_lines(right_ear, right_points)

    # Draw landmarks on left ear
    left_ear_with_landmarks = draw_landmarks_with_lines(left_ear, left_points)

    # Calculate distances between consecutive points (on LEFT ear / mirrored points)
    distances = []
    for i in range(len(left_points) - 1):
        p1 = np.array(left_points[i])
        p2 = np.array(left_points[i + 1])

        pixel_distance = float(np.linalg.norm(p1 - p2))
        cm_distance = pixel_distance / PIXELS_PER_CM

        dist_px_rounded = _round_float(pixel_distance, 3)
        dist_cm_rounded = _round_float(cm_distance, 3)

        distances.append(
            {
                "from_point": i + 1,
                "to_point": i + 2,
                "distance_pixels": float(dist_px_rounded),
                "distance_cm": float(dist_cm_rounded),
            }
        )

    # Total distance
    total_px = sum(d["distance_pixels"] for d in distances)
    total_cm = sum(d["distance_cm"] for d in distances)

    total_px_rounded = _round_float(total_px, 3)
    total_cm_rounded = _round_float(total_cm, 3)

    # Clean up session (optional - keeps memory usage low)
    # del sessions[session_id]

    # Store points in session so /validate-frame (webcam) can use them
    sessions[session_id]["right_points"] = right_points
    sessions[session_id]["left_points"]  = left_points

    logger.info(
        f"Session {session_id}: {len(right_points)} landmarks processed, "
        f"{len(distances)} distances calculated"
    )

    return JSONResponse(
        {
            "success": True,
            "data": {
                "session_id": session_id,
                "image_size": IMAGE_SIZE,
                "pixels_per_cm": PIXELS_PER_CM,
                "right_ear": {
                    "points": [{"x": x, "y": y} for x, y in right_points],
                    "landmarks_image": image_to_base64(right_ear_with_landmarks),
                },
                "left_ear": {
                    "points": [
                        {"x": x, "y": y} for x, y in left_points
                    ],
                    "landmarks_image": image_to_base64(left_ear_with_landmarks),
                },
                "distances": distances,
                "total_distance": {
                    "pixels": float(total_px_rounded),
                    "cm": float(total_cm_rounded),
                },
            },
        }
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session status"""
    if session_id in sessions:
        return {"success": True, "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/session-points/{session_id}")
async def get_session_points(session_id: str, side: str = "left"):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    points = session.get("left_points" if side == "left" else "right_points", [])
    return {"success": True, "points": points}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory"""
    if session_id in sessions:
        sessions.pop(session_id)
        return {"success": True, "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# ----------------------------------------------------------
# Standalone Live Mode Launcher
# ----------------------------------------------------------
@app.post("/start-live-validation")
async def start_live_validation(session_id: str = Form(...), side: str = Form("left")):
    """
    Launches the standalone live_validation.py script for Phase 1.
    """
    script_path = os.path.join(BASE_DIR, "live_validation.py")
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail="live_validation.py not found")

    try:
        def run_script():
            # Pass session_id and side as arguments
            subprocess.run(["python", script_path, "--session_id", session_id, "--side", side], check=True)
        
        thread = threading.Thread(target=run_script)
        thread.daemon = True
        thread.start()
        
        return {"success": True, "message": "Live validation window launched on server desktop."}
    except Exception as e:
        logger.error(f"Failed to launch live validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WEBCAM VALIDATION HELPERS ====================

def detect_blue_markers(image):
    """Return sorted list of (cx, cy) centres of blue dots in the ear image."""
    # Handle both BGR and BGRA
    if len(image.shape) == 3 and image.shape[2] == 4:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = image
        
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Detect blue markers: H: 90-130, S: 50-255, V: 50-255
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        area = cv2.contourArea(c)
        # Be very sensitive to small dots (area > 10 pixels)
        if area > 10:
            M = cv2.moments(c)
            if M["m00"] != 0:
                pts.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
    pts.sort(key=lambda p: p[1])   # top to bottom
    return pts


def get_point_guidance(digital, live, px_per_cm=PIXELS_PER_CM):
    """Return (message, error_cm) comparing a live marker to a digital point."""
    dx = digital[0] - live[0]   # +ve → live is left of target → move right
    dy = digital[1] - live[1]   # +ve → live is above target  → move down
    dist_px = float(np.hypot(dx, dy))
    tol = 5                      # pixels considered "correct"
    mm_pp = 10.0 / px_per_cm    # mm per pixel

    if dist_px <= tol:
        return "CORRECT ✓", _round_float(dist_px / px_per_cm, 3)

    hdir = ""
    vdir = ""
    if   dx >  tol: hdir = f"RIGHT {abs(dx)*mm_pp:.1f}mm"
    elif dx < -tol: hdir = f"LEFT  {abs(dx)*mm_pp:.1f}mm"
    if   dy >  tol: vdir = f"DOWN  {abs(dy)*mm_pp:.1f}mm"
    elif dy < -tol: vdir = f"UP    {abs(dy)*mm_pp:.1f}mm"

    msg = " & ".join(filter(None, [hdir, vdir]))
    return f"Move → {msg}", _round_float(dist_px / px_per_cm, 3)


@app.post("/validate-frame")
async def validate_frame(
    file: UploadFile = File(...),
    session_id: str  = Form(...),
    ear_side: str    = Form("left"),
):
    """
    Live webcam validation endpoint.
    Accepts a single JPEG frame, normalises the ear with YOLO, detects
    blue physical markers, compares each to the stored digital points,
    and returns per-point guidance + an annotated image.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    if "right_points" not in session:
        raise HTTPException(
            status_code=400,
            detail="No landmark points saved. Complete Mirror & Measure first."
        )

    digital_pts = session["left_points"] if ear_side == "left" else session["right_points"]

    # Decode incoming frame
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image frame")

    # Normalize ear
    try:
        ear = segment_and_normalize(frame)
    except Exception:
        ear = None

    if ear is None:
        return JSONResponse({"success": True, "ear_detected": False,
                             "guidance": [], "annotated_image": None})

    # Detect live blue markers
    live_pts  = detect_blue_markers(ear)
    annotated = ear.copy()

    # Draw digital target points (black ring)
    for i, dp in enumerate(digital_pts):
        cv2.circle(annotated, (int(dp[0]), int(dp[1])), 7, (0, 0, 0), 2)
        cv2.putText(annotated, str(i + 1),
                    (int(dp[0]) + 8, int(dp[1]) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)

    # Match each digital point to nearest un-used live marker
    used    = set()
    results: List[Dict[str, Any]] = []
    for i, dp in enumerate(digital_pts):
        best_d, best_j = 9999, -1
        for j, lp in enumerate(live_pts):
            if j in used: continue
            d = np.hypot(lp[0] - dp[0], lp[1] - dp[1])
            if d < best_d:
                best_d, best_j = d, j

        if best_j != -1 and best_d < IMAGE_SIZE * 0.4:
            used.add(best_j)
            matched_lp = live_pts[best_j]
            msg, err_cm = get_point_guidance(dp, matched_lp)
            correct = "CORRECT" in msg
            color   = (0, 220, 80) if correct else (255, 120, 0) # Green if OK, Blue-ish if not

            # Draw live physical marker (solid blue fill/white outline)
            cv2.circle(annotated, (int(matched_lp[0]), int(matched_lp[1])), 6, (255, 50, 50), -1)
            cv2.circle(annotated, (int(matched_lp[0]), int(matched_lp[1])), 6, (255, 255, 255), 1)
            
            # Arrow from live marker → digital target
            cv2.arrowedLine(annotated,
                            (int(matched_lp[0]), int(matched_lp[1])),
                            (int(dp[0]), int(dp[1])),
                            (0, 255, 0), 1, tipLength=0.3)
            
            # Short guidance on image
            msg_parts = msg.split("\u2192")
            guidance_part = str(msg_parts[-1]) if msg_parts else ""
            short = "OK" if correct else str(guidance_part).strip()[:14]
            cv2.putText(annotated, short,
                        (int(matched_lp[0]) + 8, int(matched_lp[1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            results.append({
                "point": i + 1, "message": msg,
                "error_cm": err_cm, "correct": correct,
                "live": [matched_lp[0], matched_lp[1]], "digital": [dp[0], dp[1]]
            })
        else:
            results.append({
                "point": i + 1, "message": "No marker detected",
                "error_cm": None, "correct": False,
                "live": None, "digital": [dp[0], dp[1]]
            })

    # Calculate overall improvement metric
    valid_points = [r for r in results if r["error_cm"] is not None]
    overall_accuracy = 0
    if valid_points:
        # 10mm (1cm) error = 0% accuracy, 0mm error = 100% accuracy
        avg_err_mm = np.mean([r["error_cm"] * 10 for r in valid_points])
        overall_accuracy = max(0.0, _round_float(100.0 - (float(avg_err_mm) * 10.0), 1)) 

    return JSONResponse({
        "success": True,
        "ear_detected": True,
        "ear_side": ear_side,
        "guidance": results,
        "summary": {
            "overall_accuracy": overall_accuracy,
            "detected_markers": len(used),
            "total_points": len(digital_pts),
            "status": "Excellent" if overall_accuracy > 90 else "Good" if overall_accuracy > 70 else "Needs Work"
        },
        "annotated_image": image_to_base64(annotated)
    })


# ==================== PRE-PIERCING DETECTION (PHASE 2) ====================

@app.post("/detect-prepiercing")
async def detect_prepiercing(
    file: UploadFile = File(...),
    session_id: str = Form(default=""),
    save_to_session: bool = Form(default=True),
):
    """
    Phase 2: Pre-Piercing Detection
    
    Analyzes an image with blue points marked on the ear to detect and count them.
    These marked blue points represent the locations where piercing will be done.
    
    Args:
        file: Image file with blue points marked on ear
        session_id: Optional session ID to save this as baseline (pre-piercing state)
        save_to_session: Whether to save the image as pre-piercing baseline
    
    Returns:
        - points_count: Number of blue points detected
        - points: List of detected point coordinates
        - annotated_image: Image with detected points highlighted and numbered
        - message: Status message
    """
    try:
        # Decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image file")
        
        # Segment ear first to ensure we are looking at the right area
        try:
            ear_normalized = segment_and_normalize(image)
        except Exception as e:
            logger.warning(f"Ear segmentation failed in pre-piercing detection: {str(e)}")
            ear_normalized = image # Fallback to original if segmentation fails
            
        # Detect blue points on the normalized ear
        points, mask = detect_blue_points(ear_normalized)
        point_count = len(points)
        
        is_prepiercing = point_count > 0
        
        if not is_prepiercing:
            logger.warning("No blue points detected. Unable to declare as pre-piercing.")
            return JSONResponse({
                "success": True,
                "is_prepiercing": False,
                "points_count": 0,
                "points": [],
                "annotated_image": image_to_base64(ear_normalized),
                "message": "⚠️ No blue points detected. This image does NOT qualify as a PRE-PIERCING state."
            })
        
        # Draw detected points on image
        annotated = draw_detected_points(ear_normalized, points)
        
        # Store pre-piercing baseline in session if requested
        prepiercing_data = {
            "points": points,
            "points_count": point_count,
            "image": ear_normalized.copy(),
            "annotated_image": annotated.copy()
        }
        
        if session_id and save_to_session and session_id in sessions:
            sessions[session_id]["prepiercing"] = prepiercing_data
            logger.info(f"Session {session_id}: Pre-piercing baseline saved with {point_count} points")
        
        # Prepare response
        response = {
            "success": True,
            "is_prepiercing": True,
            "stage": "PRE-PIERCING",
            "points_count": point_count,
            "points": [{"x": x, "y": y} for x, y in points],
            "annotated_image": image_to_base64(annotated),
            "message": f"✅ SUCCESS: {point_count} blue point(s) detected. Image declared as PRE-PIERCING BASELINE."
        }
        
        if session_id and session_id in sessions:
            response["session_id"] = session_id
            response["prepiercing_saved"] = True
        else:
            response["prepiercing_saved"] = False
        
        return JSONResponse(response)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pre-piercing detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-prepiercing-baseline")
async def get_prepiercing_baseline(session_id: str):
    """
    Get the saved pre-piercing baseline for a session.
    
    Returns:
        - points_count: Number of marked points
        - points: Coordinates of all marked points
        - baseline_image: The annotated image showing marked points
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if "prepiercing" not in session:
        raise HTTPException(
            status_code=404, 
            detail="No pre-piercing baseline found. Please run /detect-prepiercing first."
        )
    
    prepiercing = session["prepiercing"]
    
    return JSONResponse({
        "success": True,
        "session_id": session_id,
        "points_count": prepiercing["points_count"],
        "points": [{"x": x, "y": y} for x, y in prepiercing["points"]],
        "baseline_image": image_to_base64(prepiercing["annotated_image"]),
        "message": f"Pre-piercing baseline loaded: {prepiercing['points_count']} points"
    })


# ==================== LIGHTWEIGHT EAR DETECTION (AUTO-VALIDATION) ====================

@app.post("/detect-ear")
async def detect_ear(file: UploadFile = File(...)):
    """
    Lightweight endpoint: quickly checks if an ear is present in the frame.
    Uses the main YOLO model for reliability.
    """
    if model is None:
        return JSONResponse({"ear_detected": False, "error": "Model not loaded"})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"ear_detected": False, "error": "Invalid image"})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)
        # Use the main model which is already loaded
        results = model.predict(source=tmp_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        r = results[0]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    has_ear = (r.boxes is not None and len(r.boxes) > 0) or (r.masks is not None and len(r.masks) > 0)
    confidence = 0.0
    if has_ear and r.boxes is not None and len(r.boxes.conf) > 0:
        confidence = float(r.boxes.conf[0].cpu().numpy())

    return JSONResponse({
        "ear_detected": has_ear,
        "confidence": round(confidence, 3)
    })


# ==================== MOUNT UI ====================
app.mount("/ui", StaticFiles(directory=BASE_DIR, html=True), name="ui")

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
