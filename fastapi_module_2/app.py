# ==========================================================
# FASTAPI MODULE 2 - Ear Landmark Mapping API (Core)
# ==========================================================
# Features:
#   1. Upload right + left ear images → segment & normalize
#   2. User clicks landmark points on the right ear
#   3. Points are mirrored to the left ear
#   4. Distances between consecutive landmarks are calculated
#   5. Live webcam validation with blue marker guidance
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

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
# Try env var first, then common paths
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(PROJECT_ROOT, "best.pt"))
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
    description="API for ear segmentation, landmark annotation, and mirroring",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LOAD MODEL ====================
model = None
model_load_error = None
loaded_model_path = None

def get_model_candidates() -> List[str]:
    candidates = []
    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        candidates.append(os.path.abspath(env_model_path))
    
    candidates.extend([
        os.path.join(PROJECT_ROOT, "best.pt"),
        os.path.join(BASE_DIR, "best.pt"),
        os.path.join(PROJECT_ROOT, "FASTAPI", "best.pt"),
    ])
    return list(dict.fromkeys(candidates)) # Deduplicate

def load_model_on_startup():
    global model, model_load_error, loaded_model_path
    try:
        candidates = get_model_candidates()
        for path in candidates:
            if os.path.isfile(path):
                try:
                    model = YOLO(path)
                    loaded_model_path = path
                    model_load_error = None
                    logger.info(f"✓ Model loaded successfully from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load from {path}: {e}")
        
        raise FileNotFoundError(f"Model file not found. Checked: {candidates}")
    except Exception as e:
        logger.error(f"✗ Model Load Failed: {str(e)}")
        model_load_error = str(e)

load_model_on_startup()

def ensure_model_loaded():
    if model is None:
        load_model_on_startup()
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {model_load_error}")

# ==================== MODELS & UTILS ====================
class Point(BaseModel):
    x: float
    y: float

class MirrorAndMeasureRequest(BaseModel):
    right_ear_points: List[Point]
    session_id: str

def _round_float(v, n):
    return float(int(v * (10**n) + 0.5)) / (10**n)

def image_to_base64(image_array):
    rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def segment_and_normalize(image_array):
    ensure_model_loaded()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image_array)
        results = model.predict(source=tmp_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        r = results[0]
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

    if r.masks is None or len(r.masks.data) == 0:
        raise ValueError("No ear detected")

    mask = r.masks.data[0].cpu().numpy()
    mask = (mask > MASK_THRESHOLD).astype(np.uint8)
    mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_NEAREST)

    coords = np.column_stack(np.where(mask > 0))
    cy, cx = coords.mean(axis=0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    max_radius = max(max(abs(y_max - cy), abs(cy - y_min)), max(abs(x_max - cx), abs(cx - x_min)))
    max_dim = int(max_radius * 2 * 1.15)
    
    x1, y1 = int(cx - max_dim//2), int(cy - max_dim//2)
    img_h, img_w = image_array.shape[:2]
    
    cropped = image_array[max(0, y1):min(img_h, y1+max_dim), max(0, x1):min(img_w, x1+max_dim)]
    
    # Pad to square
    pad_t, pad_b = max(0, -y1), max(0, (y1+max_dim)-img_h)
    pad_l, pad_r = max(0, -x1), max(0, (x1+max_dim)-img_w)
    
    padded = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return cv2.resize(padded, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

def draw_landmarks_with_lines(image, points):
    img = image.copy()
    for i in range(len(points) - 1):
        cv2.line(img, (int(points[i][0]), int(points[i][1])), (int(points[i+1][0]), int(points[i+1][1])), (0, 255, 0), 2)
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img

# ==================== SESSION STORE ====================
sessions: Dict[str, Any] = {}

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "Ear Landmark Mapping API (Core)",
        "model_loaded": model is not None,
        "loaded_model_path": loaded_model_path,
        "endpoints": {
            "phase_1": ["/segment", "/mirror-and-measure"],
            "validation": "/validate-frame",
            "system": ["/health", "/session/{id}"]
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "loaded_model": loaded_model_path,
        "model_error": model_load_error,
        "image_size": IMAGE_SIZE
    }

@app.post("/segment")
async def segment_ears(rightEar: UploadFile = File(...), leftEar: UploadFile = File(...)):
    ensure_model_loaded()
    try:
        r_img = cv2.imdecode(np.frombuffer(await rightEar.read(), np.uint8), cv2.IMREAD_COLOR)
        l_img = cv2.imdecode(np.frombuffer(await leftEar.read(), np.uint8), cv2.IMREAD_COLOR)
        
        r_norm = segment_and_normalize(r_img)
        l_norm = segment_and_normalize(l_img)
        
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"right_ear": r_norm, "left_ear": l_norm}
        
        return {"success": True, "data": {
            "session_id": session_id,
            "right_ear_image": image_to_base64(r_norm),
            "left_ear_image": image_to_base64(l_norm)
        }}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mirror-and-measure")
async def mirror_and_measure(request: MirrorAndMeasureRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    r_pts = [(p.x, p.y) for p in request.right_ear_points]
    l_pts = [(_round_float(IMAGE_SIZE - x, 1), _round_float(y, 1)) for x, y in r_pts]
    
    distances = []
    for i in range(len(l_pts) - 1):
        px_dist = float(np.linalg.norm(np.array(l_pts[i]) - np.array(l_pts[i+1])))
        distances.append({
            "from": i+1, "to": i+2, 
            "px": _round_float(px_dist, 2), 
            "cm": _round_float(px_dist/PIXELS_PER_CM, 3)
        })
    
    session["right_points"] = r_pts
    session["left_points"] = l_pts
    
    return {"success": True, "data": {
        "right_ear": {"image": image_to_base64(draw_landmarks_with_lines(session["right_ear"], r_pts))},
        "left_ear": {"image": image_to_base64(draw_landmarks_with_lines(session["left_ear"], l_pts))},
        "distances": distances,
        "total_cm": _round_float(sum(d["cm"] for d in distances), 3)
    }}

# ==================== WEBCAM VALIDATION ====================

def detect_blue_markers(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        if cv2.contourArea(c) > 10:
            M = cv2.moments(c)
            if M["m00"] != 0:
                pts.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
    pts.sort(key=lambda p: p[1])
    return pts

@app.post("/validate-frame")
async def validate_frame(file: UploadFile = File(...), session_id: str = Form(...), ear_side: str = Form("left")):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    target_pts = session.get("left_points" if ear_side == "left" else "right_points")
    if not target_pts: raise HTTPException(status_code=400, detail="Run Mirror & Measure first")

    frame = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    try: ear = segment_and_normalize(frame)
    except: return {"success": True, "ear_detected": False}

    live_pts = detect_blue_markers(ear)
    annotated = ear.copy()
    results = []
    
    for i, dp in enumerate(target_pts):
        best_d, best_lp = 999, None
        for lp in live_pts:
            d = np.hypot(lp[0]-dp[0], lp[1]-dp[1])
            if d < best_d: best_d, best_lp = d, lp
        
        if best_lp and best_d < 80:
            err_mm = (best_d / PIXELS_PER_CM) * 10
            correct = err_mm < 1.0 # Within 1mm
            results.append({"point": i+1, "error_mm": _round_float(err_mm, 1), "correct": correct})
            cv2.circle(annotated, (int(best_lp[0]), int(best_lp[1])), 6, (255, 50, 50), -1)
            cv2.arrowedLine(annotated, (int(best_lp[0]), int(best_lp[1])), (int(dp[0]), int(dp[1])), (0, 255, 0), 1)
        else:
            results.append({"point": i+1, "error_mm": None, "correct": False})
            
    return {"success": True, "ear_detected": True, "guidance": results, "image": image_to_base64(annotated)}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        sessions.pop(session_id)
        return {"success": True}
    raise HTTPException(status_code=404, detail="Not found")

app.mount("/ui", StaticFiles(directory=BASE_DIR, html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)