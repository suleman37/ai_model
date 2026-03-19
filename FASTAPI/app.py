from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import logging
import os
from typing import List

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
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "best.pt"))
PIXELS_PER_CM = float(os.getenv("PIXELS_PER_CM", "100"))
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
    title="Ear Segmentation API",
    description="API for ear segmentation and measurement using YOLO",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ==================== LOAD MODEL ====================
def get_model_candidates() -> List[str]:
    candidates = []
    env_model_path = os.getenv("MODEL_PATH")

    if env_model_path:
        candidates.append(env_model_path)
        if not os.path.isabs(env_model_path):
            candidates.append(os.path.join(BASE_DIR, env_model_path))
            candidates.append(os.path.join(PROJECT_ROOT, env_model_path))

    candidates.extend(
        [
            os.path.join(BASE_DIR, "best.pt"),
            os.path.join(PROJECT_ROOT, "FASTAPI", "best.pt"),
            os.path.join(PROJECT_ROOT, "best.pt"),
        ]
    )

    normalized_candidates = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_candidates.append(normalized)

    return normalized_candidates


def load_model_on_startup():
    global model, model_load_error, loaded_model_path
    try:
        ensure_runtime_dependencies()

        available_candidates = get_model_candidates()
        existing_candidates = [path for path in available_candidates if os.path.isfile(path)]

        if not existing_candidates:
            raise FileNotFoundError(
                "No model file found. Checked: " + ", ".join(available_candidates)
            )

        load_errors = []
        for candidate in existing_candidates:
            try:
                model = YOLO(candidate)
                loaded_model_path = candidate
                model_load_error = None
                logger.info(f"✓ Model loaded successfully from {candidate}")
                return
            except Exception as candidate_error:
                load_errors.append(f"{candidate}: {candidate_error}")

        raise RuntimeError(
            "Failed to load model from available files. Errors: " + " | ".join(load_errors)
        )
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        model = None
        loaded_model_path = None
        model_load_error = str(e)

model = None
model_load_error = None
loaded_model_path = None
load_model_on_startup()


def ensure_model_loaded() -> None:
    if model is None:
        load_model_on_startup()

    if model is None:
        detail = model_load_error or "Model not loaded"
        raise HTTPException(status_code=500, detail=detail)

# ==================== UTILITY FUNCTIONS ====================

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    ensure_runtime_dependencies()
    pil_image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64


def process_ear_segmentation(image_array, confidence=CONFIDENCE_THRESHOLD):
    """
    Process ear segmentation and extract measurements
    Returns: (measurements, images)
    """
    ensure_runtime_dependencies()

    if model is None:
        raise Exception("Model not loaded")
    
    # Run prediction
    results = model.predict(source=image_array, conf=confidence, verbose=False)
    r = results[0]
    
    # Convert to RGB if needed
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        image = image_array
    
    # Get mask
    if r.masks is None:
        raise ValueError("No ear detected in the image")
    
    mask = r.masks.data[0].cpu().numpy()
    mask = (mask > MASK_THRESHOLD).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # ===== OUTPUT 1: Segmentation Result =====
    segmentation_result = r.plot()
    segmentation_result_rgb = cv2.cvtColor(segmentation_result, cv2.COLOR_BGR2RGB)
    
    # ===== OUTPUT 2: Segmented Ear (Black Background) =====
    segmented_ear = image.copy()
    segmented_ear[mask == 0] = 0
    
    # ===== OUTPUT 3: Mask Visualization =====
    mask_visualization = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
    
    # ===== MEASUREMENTS =====
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0:
        raise ValueError("No valid segmentation mask")
    
    ear_height = coords[:, 0].max() - coords[:, 0].min()
    ear_width = coords[:, 1].max() - coords[:, 1].min()
    area = np.sum(mask)
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found")
    
    cnt = contours[0]
    perimeter = cv2.arcLength(cnt, True)
    aspect_ratio = ear_width / ear_height if ear_height != 0 else 0
    
    ellipse = cv2.fitEllipse(cnt)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    
    # Convert pixels to cm
    ear_height_cm = ear_height / PIXELS_PER_CM
    ear_width_cm = ear_width / PIXELS_PER_CM
    perimeter_cm = perimeter / PIXELS_PER_CM
    major_axis_cm = major_axis / PIXELS_PER_CM
    minor_axis_cm = minor_axis / PIXELS_PER_CM
    area_cm2 = area / (PIXELS_PER_CM ** 2)
    
    measurements = {
        "ear_height_px": int(ear_height),
        "ear_width_px": int(ear_width),
        "ear_height_cm": round(ear_height_cm, 2),
        "ear_width_cm": round(ear_width_cm, 2),
        "area_px2": int(area),
        "area_cm2": round(area_cm2, 2),
        "perimeter_px": round(perimeter, 2),
        "perimeter_cm": round(perimeter_cm, 2),
        "aspect_ratio": round(aspect_ratio, 4),
        "major_axis_px": round(major_axis, 2),
        "major_axis_cm": round(major_axis_cm, 2),
        "minor_axis_px": round(minor_axis, 2),
        "minor_axis_cm": round(minor_axis_cm, 2),
    }
    
    images = {
        "segmentation_result": image_to_base64(segmentation_result_rgb),
        "segmented_ear": image_to_base64(segmented_ear),
        "mask_visualization": image_to_base64(mask_visualization),
    }
    
    return measurements, images

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ear Segmentation API",
        "status": "running",
        "model_loaded": model is not None,
        "loaded_model_path": loaded_model_path,
        "model_error": model_load_error,
        "endpoints": {
            "health": "/health",
            "predict_full": "/predict",
            "predict_simple": "/predict-simple",
            "predict_dual": "/predict-dual",
            "demo": "/demo",
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "configured_model_path": MODEL_PATH,
        "loaded_model_path": loaded_model_path,
        "available_model_paths": get_model_candidates(),
        "model_error": model_load_error,
        "pixels_per_cm": PIXELS_PER_CM
    }


@app.get("/demo")
async def demo():
    """Serve the bundled frontend demo."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...), confidence: float = CONFIDENCE_THRESHOLD):
    """
    Full prediction with all measurements and 3 output images
    
    Returns:
    - All measurements (height, width, area, perimeter, aspect ratio, axes)
    - 3 visualizations: segmentation result, segmented ear, mask visualization
    """
    
    try:
        ensure_runtime_dependencies()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    ensure_model_loaded()
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image file")
        
        measurements, images = process_ear_segmentation(image, confidence=confidence)
        
        return JSONResponse({
            "success": True,
            "data": {
                "measurements": measurements,
                "images": images,
                "calibration": {
                    "pixels_per_cm": PIXELS_PER_CM
                }
            }
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict-simple")
async def predict_simple(file: UploadFile = File(...)):
    """
    Simple prediction - Returns ONLY ear height and width (primary parameters)
    
    Returns:
    - ear_height_cm & ear_height_px (vertical)
    - ear_width_cm & ear_width_px (horizontal)
    - 3 output images
    """
    
    try:
        ensure_runtime_dependencies()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    ensure_model_loaded()
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image file")
        
        measurements, images = process_ear_segmentation(image)
        
        return JSONResponse({
            "success": True,
            "data": {
                "height": {
                    "pixels": measurements["ear_height_px"],
                    "cm": measurements["ear_height_cm"]
                },
                "width": {
                    "pixels": measurements["ear_width_px"],
                    "cm": measurements["ear_width_cm"]
                },
                "images": images
            }
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict-dual")
async def predict_dual(imageLeft: UploadFile = File(...), imageRight: UploadFile = File(...)):
    """
    Dual prediction - Process LEFT ear and RIGHT ear images
    
    Returns:
    - Results for left ear (height & width in pixels & cm)
    - Results for right ear (height & width in pixels & cm)
    - 3 output images for each ear (6 total)
    """
    
    try:
        ensure_runtime_dependencies()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    ensure_model_loaded()
    
    try:
        # Process LEFT image
        left_contents = await imageLeft.read()
        left_nparr = np.frombuffer(left_contents, np.uint8)
        left_image = cv2.imdecode(left_nparr, cv2.IMREAD_COLOR)
        
        if left_image is None:
            raise ValueError("Invalid left image file")
        
        left_measurements, left_images = process_ear_segmentation(left_image)
        
        # Process RIGHT image
        right_contents = await imageRight.read()
        right_nparr = np.frombuffer(right_contents, np.uint8)
        right_image = cv2.imdecode(right_nparr, cv2.IMREAD_COLOR)
        
        if right_image is None:
            raise ValueError("Invalid right image file")
        
        right_measurements, right_images = process_ear_segmentation(right_image)
        
        return JSONResponse({
            "success": True,
            "data": {
                "left_ear": {
                    "height": {
                        "pixels": left_measurements["ear_height_px"],
                        "cm": left_measurements["ear_height_cm"]
                    },
                    "width": {
                        "pixels": left_measurements["ear_width_px"],
                        "cm": left_measurements["ear_width_cm"]
                    },
                    "images": left_images
                },
                "right_ear": {
                    "height": {
                        "pixels": right_measurements["ear_height_px"],
                        "cm": right_measurements["ear_height_cm"]
                    },
                    "width": {
                        "pixels": right_measurements["ear_width_px"],
                        "cm": right_measurements["ear_width_cm"]
                    },
                    "images": right_images
                },
                "comparison": {
                    "height_difference_cm": round(abs(left_measurements["ear_height_cm"] - right_measurements["ear_height_cm"]), 2),
                    "width_difference_cm": round(abs(left_measurements["ear_width_cm"] - right_measurements["ear_width_cm"]), 2),
                    "height_difference_px": abs(left_measurements["ear_height_px"] - right_measurements["ear_height_px"]),
                    "width_difference_px": abs(left_measurements["ear_width_px"] - right_measurements["ear_width_px"])
                }
            }
        })
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Dual prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
