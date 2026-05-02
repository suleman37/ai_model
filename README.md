# 👂 Ear Piercing Landmark Mapping System

> A computer-vision-powered FastAPI system for precise ear piercing planning — using a custom-trained YOLOv8 segmentation model to isolate ears, mirror landmarks bilaterally, measure distances, and validate physical marker placement in real time via webcam.

---

## 🗂️ Project Structure

```
p/
├── best.pt                      # 🤖 Shared YOLOv8 segmentation model
├── phase 1/
│   ├── app.py                   # ⚡ Phase 1 FastAPI server (port 8001)
│   ├── blue_point_detector.py   # 🔵 Basic blue-dot detection utility
│   ├── index.html               # 🌐 Main frontend UI
│   └── blue_detector.html       # 🔬 Blue marker preview tool
├── phase 2/
│   ├── app.py                   # ⚡ Phase 2 FastAPI server (port 8080)
│   ├── blue_point_detector.py   # 🔵 Improved detector (morphology + circularity)
│   └── blue_detector.html       # 🔬 Updated blue marker preview tool
└── phase 3/
    ├── app.py                   # ⚡ Phase 3 FastAPI server (port 8080)
    ├── blue_point_detector.py   # 🔵 Same improved detector as Phase 2
    └── index.html               # 🌐 Full-featured SPA with piercing-type support
```

---

## 🔄 End-to-End Flow

```
📸 Upload R+L Ear Images
        ↓
🤖 YOLO Segmentation
        ↓
📐 Normalize to 256×256
        ↓
🖱️ User Clicks Landmarks on Right Ear
        ↓
↔️ Horizontal Mirror → Left Ear Points
        ↓
📏 Distance Measurement (pixels + cm)
        ↓
📷 Live Webcam Validation
   (Physical blue markers ↔ Digital targets)
```

---

## 📐 Phase 1 — Core Landmark Mapping

**File:** `phase 1/app.py` · **Port:** `8001`

Phase 1 establishes the foundational pipeline. The system accepts uploads of both the right and left ear images, runs YOLOv8 instance segmentation on each, crops a square around the ear's center of mass with 15% padding, and resizes to a canonical **256×256** image. A session ID is issued to tie all subsequent requests together.

The user then clicks landmark points on the rendered right ear image in the browser. When submitted, the API mirrors each point horizontally (`mirrored_x = 256 - x`) to produce corresponding left-ear coordinates, then draws numbered landmarks with connecting lines on both ear images. Euclidean distances between consecutive landmarks are computed in both pixels and centimetres (assuming **100 px/cm**).

Phase 1 also includes a pre-piercing detection endpoint (`/detect-prepiercing`) that accepts an image with physical blue marker dots already drawn on the skin, runs the blue-point detector to count and localise them, and stores the result as a "pre-piercing baseline" on the session. The first version of `blue_point_detector.py` uses a simple HSV threshold (H 90–130, S/V 50–255) with a minimum-area filter of 10 px².

### 🛣️ Phase 1 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/segment` | Upload right + left ear → normalise, create session |
| `POST` | `/mirror-and-measure` | Submit right-ear landmarks → mirror + measure |
| `POST` | `/validate-frame` | Live webcam frame → validate physical markers |
| `POST` | `/detect-prepiercing` | Detect pre-piercing blue markers, save as baseline |
| `POST` | `/get-prepiercing-baseline` | Retrieve saved baseline for a session |
| `GET`  | `/session/{id}` | Check session status |
| `DELETE` | `/session/{id}` | Remove session from memory |
| `GET`  | `/health` | Model load status + configuration |

---

## 🔵 Phase 2 — Blue Point Validation & Improved Detection

**File:** `phase 2/app.py` · **Port:** `8080`

Phase 2 refines the blue-marker detection pipeline and makes blue-point checking a first-class part of the segmentation step. After normalising both ears, the `/segment` endpoint immediately runs `detect_blue_points` on each image and reports how many blue markers are present (`right_blue_points`, `left_blue_points`, `has_blue_points`).

The blue-point detector in Phase 2 is significantly improved over Phase 1:

- ✅ Tighter HSV range (H 95–130, S 100–255, V 70–255) — reduces skin-tone false positives
- ✅ Minimum area raised to 20 px²
- ✅ Morphological opening + closing — suppresses hair noise and fills ink gaps
- ✅ Circularity check (4πA/P² ≥ 0.2) — discards jagged non-circular blobs
- ✅ Same refined range used in live-frame validator (`detect_blue_markers_live`)

### 🛣️ Phase 2 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/segment` | Upload + normalise + blue-point check |
| `POST` | `/mirror-and-measure` | Mirror landmarks + distances |
| `POST` | `/validate-frame` | Webcam validation with improved detector |
| `GET`  | `/session/{id}` | Check session |
| `DELETE` | `/session/{id}` | Delete session |

---

## 💎 Phase 3 — Piercing-Type Pattern Support

**File:** `phase 3/app.py` · **Port:** `8080`

Phase 3 adds awareness of named piercing patterns so that landmark visualisation adapts to the specific look being planned. The `MirrorRequest` model gains a `piercing_type` field and the drawing logic branches on it.

### 💅 Supported Piercing Types

| Type | Emoji | Behaviour |
|------|-------|-----------|
| `triangle` | 🔺 | Lines connect 1→2→3→1 (closed loop); perimeter distance included |
| `snakebite` | 🐍 | Gold/silver stud rendering, no connecting lines |
| `impuria` | ✨ | Stud rendering, labels suppressed |
| `Lobe | 〰️ | SImple point at the centre of lobe|
| `lobetrio` | ⭕ | Three graduated stud sizes (8 px, 6 px, 4 px) — Large → Medium → Small |

The label side (left or right of the dot) is automatically chosen based on where landmarks sit relative to the image centre, and is flipped for the mirrored left ear so labels always appear on the outer edge.

The `/segment` endpoint now also returns the raw coordinates of any detected blue points (`right_blue_points_coords`), not just the count. The root `/` serves `index.html` directly via `FileResponse`.

### 🛣️ Phase 3 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/segment` | Upload + normalise + blue-point coords |
| `POST` | `/mirror-and-measure` | Mirror + measure with piercing-type visualisation |
| `POST` | `/validate-frame` | Webcam validation |
| `GET`  | `/session/{id}` | Check session |
| `DELETE` | `/session/{id}` | Delete session |

---

## 🔧 Shared Components

### 🤖 YOLO Model (`best.pt`)
Custom-trained YOLOv8 instance-segmentation model. Loaded once at startup and shared across all inference calls. Confidence threshold: `0.25`, mask threshold: `0.5`. Input images are temporarily saved to disk for prediction and immediately deleted after inference.

### 📐 Normalisation Pipeline
All three phases use the same algorithm:
1. Run YOLO to obtain the ear segmentation mask
2. Find the bounding box and centroid of the masked region
3. Compute the maximum dimension from centroid to any edge, add **15% padding**
4. Crop a square around the centroid; replicate-pad if the crop extends outside image bounds
5. Resize to **256×256** with bilinear interpolation

### 🗃️ Session Store
All sessions are kept in an in-process Python dictionary. Each session holds the normalised ear images and, after landmarks are submitted, the digital point coordinates for both ears. Sessions persist for the lifetime of the server process.

### 📏 Pixel-to-CM Conversion
All phases assume **100 pixels = 1 cm** (`PIXELS_PER_CM = 100`). This is a fixed constant — real-world accuracy depends on consistent image capture distance and scale.

---

## 🚀 Running a Phase

```bash
# Install dependencies
pip install fastapi uvicorn ultralytics opencv-python pillow numpy

# ▶️ Phase 1
cd "phase 1"
python app.py          # http://localhost:8001

# ▶️ Phase 2
cd "phase 2"
python app.py          # http://localhost:8080

# ▶️ Phase 3
cd "phase 3"
python app.py          # http://localhost:8080
```

The frontend is served at `/ui` (Phases 1 & 2) or `/` (Phase 3).

---

## 🔮 Future Work

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ⚡ API Framework | FastAPI + Uvicorn |
| 🤖 Computer Vision | OpenCV, YOLOv8 (Ultralytics) |
| 🖼️ Image Processing | Pillow, NumPy |
| 🌐 Frontend | Vanilla HTML/JS (served as static files) |
| 🧠 Model Format | PyTorch (`.pt`) |
