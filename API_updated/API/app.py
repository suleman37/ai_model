# ==========================================================
# API UPDATED - Ear Landmark Validation API
# ==========================================================
# Extends the landmark mapping workflow with live frame validation:
#   1. Upload right + left ear images -> segment & normalize
#   2. User clicks landmark points on the right ear
#   3. Points are mirrored to the left ear
#   4. Live frames are validated against saved landmark positions
# ==========================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple
import io
import base64
import tempfile
import os
import logging
import json

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(PROJECT_ROOT, "FASTAPI", "best.pt"))
PIXELS_PER_CM = float(os.getenv("PIXELS_PER_CM", "100"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
MASK_THRESHOLD = float(os.getenv("MASK_THRESHOLD", "0.5"))

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_runtime_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if cv2 is None:
        missing.append("opencv-python")
    if Image is None:
        missing.append("pillow")
    if YOLO is None:
        missing.append("ultralytics")

    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install them in the active virtualenv to use inference endpoints."
        )

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Ear Landmark Validation API",
    description="API for ear segmentation, landmark annotation, mirroring, distance measurement, and live frame validation",
    version="3.0.0",
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


def load_model_on_startup():
    global model
    try:
        ensure_runtime_dependencies()
        model = YOLO(MODEL_PATH)
        logger.info(f"✓ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        model = None


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


def image_to_base64(image_array):
    """Convert numpy array (BGR) to base64 PNG string"""
    ensure_runtime_dependencies()
    # Convert BGR to RGB for PIL
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        rgb = image_array
    pil_image = Image.fromarray(rgb)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def center_ear(image, mask):
    """Center the ear in the image using the mask centroid"""
    ensure_runtime_dependencies()
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
    ensure_runtime_dependencies()

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

    # Segment ear (black background)
    segmented = image_array.copy()
    segmented[mask == 0] = 0

    # Crop ear using mask bounding box
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        raise ValueError("No valid mask pixels found")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = segmented[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Normalize to IMAGE_SIZE x IMAGE_SIZE
    normalized = cv2.resize(
        cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    mask_resized = cv2.resize(
        cropped_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
    )

    # Center ear
    normalized = center_ear(normalized, mask_resized)

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
sessions = {}


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
            "segment": "/segment (POST) - Upload right & left ear images",
            "mirror_and_measure": "/mirror-and-measure (POST) - Send right ear landmarks",
            "validate_frame": "/validate-frame (POST) - Validate a live frame against saved points",
            "session_status": "/session/{session_id} (GET) - Check session availability",
            "delete_session": "/session/{session_id} (DELETE) - Remove saved session data",
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
    try:
        ensure_runtime_dependencies()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

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
        import uuid
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
    try:
        ensure_runtime_dependencies()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

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

    if len(right_points) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 landmark points are required for distance calculation",
        )

    # Mirror points for left ear (horizontal flip)
    left_points = []
    for x, y in right_points:
        mirrored_x = (IMAGE_SIZE - 1) - int(x)
        mirrored_y = int(y)
        left_points.append((mirrored_x, mirrored_y))

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

        distances.append(
            {
                "from_point": i + 1,
                "to_point": i + 2,
                "distance_pixels": round(pixel_distance, 3),
                "distance_cm": round(cm_distance, 3),
            }
        )

    # Total distance
    total_px = sum(d["distance_pixels"] for d in distances)
    total_cm = sum(d["distance_cm"] for d in distances)

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
                    "pixels": round(total_px, 3),
                    "cm": round(total_cm, 3),
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


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory"""
    if session_id in sessions:
        del sessions[session_id]
        return {"success": True, "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# ==================== WEBCAM VALIDATION HELPERS ====================

def detect_blue_markers(image):
    """Return sorted list of (cx, cy) centres of blue dots in the ear image."""
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
        return "CORRECT ✓", round(dist_px / px_per_cm, 3)

    hdir = ""
    vdir = ""
    if   dx >  tol: hdir = f"RIGHT {abs(dx)*mm_pp:.1f}mm"
    elif dx < -tol: hdir = f"LEFT  {abs(dx)*mm_pp:.1f}mm"
    if   dy >  tol: vdir = f"DOWN  {abs(dy)*mm_pp:.1f}mm"
    elif dy < -tol: vdir = f"UP    {abs(dy)*mm_pp:.1f}mm"

    msg = " & ".join(filter(None, [hdir, vdir]))
    return f"Move → {msg}", round(dist_px / px_per_cm, 3)


@app.post("/validate-frame")
async def validate_frame(
    file: UploadFile = File(...),
    session_id: str  = Form(...),
    ear_side: str    = Form("left"),
):
    """
    Live webcam validation endpoint.
    Accepts a single JPEG frame, normalises the ear with YOLO, detects
    red physical markers, compares each to the stored digital points,
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
    results = []
    for i, dp in enumerate(digital_pts):
        best_d, best_j = 9999, -1
        for j, lp in enumerate(live_pts):
            if j in used: continue
            d = np.hypot(lp[0] - dp[0], lp[1] - dp[1])
            if d < best_d:
                best_d, best_j = d, j

        if best_j != -1 and best_d < IMAGE_SIZE * 0.4:
            used.add(best_j)
            lp      = live_pts[best_j]
            msg, err_cm = get_point_guidance(dp, lp)
            correct = "CORRECT" in msg
            color   = (0, 220, 80) if correct else (255, 120, 0) # Green if OK, Blue-ish if not

            # Draw live physical marker (solid blue fill/white outline)
            cv2.circle(annotated, (int(lp[0]), int(lp[1])), 6, (255, 50, 50), -1)
            cv2.circle(annotated, (int(lp[0]), int(lp[1])), 6, (255, 255, 255), 1)
            
            # Arrow from live marker → digital target
            cv2.arrowedLine(annotated,
                            (int(lp[0]), int(lp[1])),
                            (int(dp[0]), int(dp[1])),
                            (0, 255, 0), 1, tipLength=0.3)
            
            # Short guidance on image
            short = "OK" if correct else msg.split("→")[-1].strip()[:14]
            cv2.putText(annotated, short,
                        (int(lp[0]) + 8, int(lp[1]) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            results.append({
                "point": i + 1, "message": msg,
                "error_cm": err_cm, "correct": correct,
                "live": [lp[0], lp[1]], "digital": [dp[0], dp[1]]
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
        overall_accuracy = max(0, round(100 - (avg_err_mm * 10), 1)) 

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


# ==================== MOUNT UI ====================
app.mount("/ui", StaticFiles(directory=BASE_DIR, html=True), name="ui")

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
    try:
        ensure_runtime_dependencies()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
