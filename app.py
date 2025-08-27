#!/usr/bin/env python3
"""
DefectDetect: Clasificación simple de defectos con OpenCV + scikit-learn
- UI web para capturar desde la webcam (backend), etiquetar (BUENA/MALA), entrenar y clasificar.
- API REST (FastAPI): /predict, /feedback_webcam, /train, /camera/start|stop|status

Requisitos:
  pip install opencv-python fastapi uvicorn gradio scikit-learn numpy pydantic pillow joblib

Ejecución:
  python app.py
  # UI:   http://127.0.0.1:7860/ui
  # Docs: http://127.0.0.1:7860/docs
"""
from __future__ import annotations

import os
import time
import uuid
import json
import threading
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import gradio as gr

# ==========================
# Configuración
# ==========================
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
GOOD_DIR = DATA_DIR / "good"
BAD_DIR = DATA_DIR / "bad"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
GOOD_DIR.mkdir(parents=True, exist_ok=True)
BAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.pkl"

# HOG params (OpenCV)
WIN_SIZE = (128, 128)
BLOCK_SIZE = (32, 32)
BLOCK_STRIDE = (16, 16)
CELL_SIZE = (16, 16)
NBINS = 9

# ==========================
# Webcam (gestor persistente)
# ==========================
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))

class CameraManager:
    def __init__(self, index:int=CAMERA_INDEX, width:int=CAMERA_WIDTH, height:int=CAMERA_HEIGHT):
        self.index = index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        with self.lock:
            if self.running:
                return
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(self.index)
            if not self.cap or not self.cap.isOpened():
                self.cap = None
                raise RuntimeError(f"No se pudo abrir la cámara index={self.index}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            self.running = True

    def stop(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.running = False

    def read(self, require_running: bool = False) -> np.ndarray:
        """
        Lee un frame. Si require_running=True y la cámara está apagada, lanza RuntimeError.
        Si require_running=False y la cámara está apagada, intenta autoarrancar.
        """
        with self.lock:
            if not self.running or self.cap is None:
                if require_running:
                    raise RuntimeError("La cámara está apagada (backend no iniciado).")
                # auto-start flexible
                self.start()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("Fallo al leer frame de la cámara.")
            return frame

cam = CameraManager()

# ==========================
# Última foto + resultado (estado compartido UI/API)
# ==========================
LAST_FRAME: Optional[np.ndarray] = None
LAST_VERSION: int = 0
LAST_RESULT: Optional[Dict] = None
_LAST_LOCK = threading.Lock()

def set_last_frame(frame: np.ndarray) -> None:
    global LAST_FRAME, LAST_VERSION
    with _LAST_LOCK:
        LAST_FRAME = frame.copy()
        LAST_VERSION += 1

def get_last_frame() -> Optional[np.ndarray]:
    with _LAST_LOCK:
        return None if LAST_FRAME is None else LAST_FRAME.copy()

def get_last_version() -> int:
    with _LAST_LOCK:
        return LAST_VERSION

def set_last_result(res: Dict) -> None:
    global LAST_RESULT
    with _LAST_LOCK:
        LAST_RESULT = dict(res)

def get_last_result() -> Optional[Dict]:
    with _LAST_LOCK:
        return None if LAST_RESULT is None else dict(LAST_RESULT)

# ==========================
# Utilidades de imagen
# ==========================
def imread_any(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen (jpg/png).")
    return img

def ensure_3ch(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = ensure_3ch(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, WIN_SIZE, interpolation=cv2.INTER_AREA)

    hog = cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)
    hog_feat = hog.compute(gray_resized).reshape(-1)

    hist = cv2.calcHist([gray_resized], [0], None, [32], [0, 256]).reshape(-1)
    hist = hist / (hist.sum() + 1e-8)

    feats = np.concatenate([hog_feat, hist]).astype(np.float32)
    return feats

def capture_snapshot(flush:int=5, delay_ms:int=30, require_running:bool=True, timeout_ms:int=1500) -> np.ndarray:
    """
    Vacía el buffer leyendo varios frames y devuelve el último (fresco).
    - require_running=True: error inmediato si cámara apagada (no autoarrancar).
    - timeout_ms: corta si el driver no entrega frame en tiempo.
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    frame = None
    for i in range(max(1, flush)):
        if time.monotonic() > deadline:
            raise TimeoutError("Timeout capturando frame de la cámara.")
        frame = cam.read(require_running=require_running)
        if i < flush - 1 and delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
    return frame

# ==========================
# Modelo
# ==========================
def load_model() -> Optional[Pipeline]:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

def save_model(model: Pipeline) -> None:
    joblib.dump(model, MODEL_PATH)

def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000))
    ])

def dataset_iter() -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for p in GOOD_DIR.glob("**/*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            pairs.append((str(p), 1))
    for p in BAD_DIR.glob("**/*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            pairs.append((str(p), 0))
    return pairs

def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    pairs = dataset_iter()
    X, y = [], []
    for path, label in pairs:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        X.append(extract_features(img))
        y.append(label)
    if not X:
        raise RuntimeError("El dataset está vacío. Etiqueta algunas imágenes primero.")
    return np.vstack(X), np.array(y, dtype=np.int32)

def train_model() -> Dict:
    X, y = load_dataset()
    model = build_model()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    ypred = model.predict(Xte)

    acc = round(float(accuracy_score(yte, ypred)), 2)
    cm = confusion_matrix(yte, ypred).tolist()
    report = classification_report(yte, ypred, target_names=["MALA", "BUENA"], output_dict=True)
    for k, v in report.items():
        if isinstance(v, dict):
            for kk in v:
                try:
                    v[kk] = round(float(v[kk]), 2)
                except Exception:
                    pass

    save_model(model)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "classes": {"0": "MALA", "1": "BUENA"}}

def predict_image(img_bgr: np.ndarray) -> Dict:
    model = load_model()
    if model is None:
        raise RuntimeError("No hay modelo entrenado. Entrena primero.")
    feats = extract_features(img_bgr).reshape(1, -1)
    if hasattr(model.named_steps["lr"], "predict_proba"):
        prob_good = float(model.predict_proba(feats)[0, 1])
    else:
        dec = model.decision_function(feats)
        prob_good = float((dec - dec.min()) / (dec.max() - dec.min() + 1e-8))
    label = "BUENA" if prob_good >= 0.5 else "MALA"
    return {"label": label, "prob_good": round(prob_good, 2), "prob_bad": round(1.0 - prob_good, 2)}

# ==========================
# Guardado de ejemplos
# ==========================
def save_labeled_image(img_bgr: np.ndarray, label: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{uuid.uuid4().hex[:8]}.jpg"
    outdir = GOOD_DIR if label.upper() in {"BUENA","GOOD","1"} else BAD_DIR
    path = outdir / fname
    cv2.imwrite(str(path), img_bgr)
    return str(path)

# ==========================
# API FastAPI
# ==========================
app = FastAPI(title="DefectDetect API", version="1.7.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    label: str
    prob_good: float
    prob_bad: float

@app.post("/predict", response_model=PredictResponse)
async def api_predict(file: UploadFile = File(None)):
    """
    Clasifica una imagen.
    - Si se envía `file`, clasifica ese archivo.
    - Si `file` es None, toma snapshot de la webcam persistente (requiere cámara encendida).
    """
    try:
        if file is None:
            img = capture_snapshot(require_running=True)
            set_last_frame(img)   # para sincronizar UI
        else:
            data = await file.read()
            img = imread_any(data)
    except RuntimeError as e: raise HTTPException(status_code=409, detail=f"camera_off: {e}")
    except TimeoutError as e: raise HTTPException(status_code=504, detail=str(e))
    res = predict_image(img)
    set_last_result(res)        # para sincronizar UI
    return JSONResponse(res)

@app.post("/predict_webcam", response_model=PredictResponse)
async def api_predict_webcam():
    try:
        img = capture_snapshot(require_running=True)
        set_last_frame(img)     # para sincronizar UI
    except RuntimeError as e: raise HTTPException(status_code=409, detail=f"camera_off: {e}")
    except TimeoutError as e: raise HTTPException(status_code=504, detail=str(e))
    res = predict_image(img)
    set_last_result(res)        # para sincronizar UI
    return JSONResponse(res)

@app.post("/feedback_webcam")
async def api_feedback_webcam(label: str = Form(...)):
    img = capture_snapshot(require_running=True)
    set_last_frame(img)
    path = save_labeled_image(img, label)
    return {"saved": True, "path": path}

@app.post("/train")
async def api_train():
    try:
        return {"trained": True, **train_model()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"trained": False, "error": str(e)})

@app.post("/camera/start")
async def api_camera_start(): cam.start(); return {"running": True}

@app.post("/camera/stop")
async def api_camera_stop(): cam.stop(); return {"running": False}

@app.get("/camera/status")
async def api_camera_status(): return {"running": cam.running}

# ==========================
# UI con Gradio (mantiene funciones de todos los botones)
# ==========================
def ui_take_photo_backend() -> Tuple[str, np.ndarray]:
    """Solo captura; no clasifica."""
    try:
        frame = capture_snapshot(require_running=True)
        set_last_frame(frame)
        return "Foto capturada.", bgr_to_rgb(frame)
    except RuntimeError as e:
        return f"Error: cámara apagada. {e}", np.zeros((10,10,3), dtype=np.uint8)
    except TimeoutError as e:
        return f"Error: timeout ({e})", np.zeros((10,10,3), dtype=np.uint8)

def ui_classify_backend() -> Tuple[str, Dict[str, float], np.ndarray]:
    """Captura y clasifica; actualiza todo."""
    try:
        frame_bgr = capture_snapshot(require_running=True)
        set_last_frame(frame_bgr)
        r = predict_image(frame_bgr)
        set_last_result(r)
        return f"Resultado: {r['label']}", {"BUENA": r["prob_good"], "MALA": r["prob_bad"]}, bgr_to_rgb(frame_bgr)
    except RuntimeError as e:
        return f"Error: cámara apagada. {e}", {}, np.zeros((10,10,3), dtype=np.uint8)
    except TimeoutError as e:
        return f"Error: timeout ({e})", {}, np.zeros((10,10,3), dtype=np.uint8)

def ui_save_good_backend() -> Tuple[str, np.ndarray]:
    """Guarda última foto como BUENA; no clasifica."""
    try:
        frame = get_last_frame()
        if frame is None:
            frame = capture_snapshot(require_running=True)
            set_last_frame(frame)
        path = save_labeled_image(frame, "BUENA")
        return f"Guardada como BUENA: {path}", bgr_to_rgb(frame)
    except RuntimeError as e:
        return f"Error: cámara apagada. {e}", np.zeros((10,10,3), dtype=np.uint8)
    except TimeoutError as e:
        return f"Error: timeout ({e})", np.zeros((10,10,3), dtype=np.uint8)

def ui_save_bad_backend() -> Tuple[str, np.ndarray]:
    """Guarda última foto como MALA; no clasifica."""
    try:
        frame = get_last_frame()
        if frame is None:
            frame = capture_snapshot(require_running=True)
            set_last_frame(frame)
        path = save_labeled_image(frame, "MALA")
        return f"Guardada como MALA: {path}", bgr_to_rgb(frame)
    except RuntimeError as e:
        return f"Error: cámara apagada. {e}", np.zeros((10,10,3), dtype=np.uint8)
    except TimeoutError as e:
        return f"Error: timeout ({e})", np.zeros((10,10,3), dtype=np.uint8)

def ui_train() -> str:
    try:
        m = train_model()
        return (
            "Entrenado. Accuracy: {acc:.2f}. Train/Test: {ntr}/{nte}\n".format(
                acc=m["accuracy"], ntr=m["n_train"], nte=m["n_test"]
            )
            + "Matriz de confusión: "
            + json.dumps(m["confusion_matrix"])
        )
    except Exception as e:
        return f"Error al entrenar: {e}"

# ---- Timer: refresco automático si API externa tomó una nueva foto y predijo ----
def ui_pull_last(prev_version: int):
    """
    Si hay nueva foto (versión distinta), devuelve:
      - nueva versión
      - imagen RGB actualizada
      - label de probabilidades
      - texto de resultado
    Si no hay cambios, devuelve updates vacíos para no tocar la UI.
    """
    try:
        ver = get_last_version()
        if ver != prev_version:
            frame = get_last_frame()
            res = get_last_result()
            img_update = gr.update(value=bgr_to_rgb(frame)) if frame is not None else gr.update()
            if res:
                lbl_update = gr.update(value={"BUENA": res["prob_good"], "MALA": res["prob_bad"]})
                txt_update = gr.update(value=f"Resultado: {res['label']}")
            else:
                lbl_update, txt_update = gr.update(), gr.update()
            return ver, img_update, lbl_update, txt_update
        else:
            return prev_version, gr.update(), gr.update(), gr.update()
    except Exception:
        return prev_version, gr.update(), gr.update(), gr.update()

with gr.Blocks(title="DefectDetect UI") as demo:
    gr.Markdown("# DefectDetect\nBackend: captura, etiqueta, entrena y clasifica.")

    with gr.Row():
        out_img = gr.Image(label="Última captura (backend)", interactive=False)
        lbl = gr.Label(label="Probabilidades")
    out_text = gr.Textbox(label="Estado / Resultado", interactive=False)

    ver_state = gr.State(0)

    with gr.Row():
        btn_on   = gr.Button("Encender cámara (backend)")
        btn_off  = gr.Button("Apagar cámara (backend)")
        btn_snap = gr.Button("Tomar foto (backend)")
        btn_pred = gr.Button("Clasificar (tomar foto y clasificar)")
        btn_good = gr.Button("Marcar BUENA ✅ (última foto)")
        btn_bad  = gr.Button("Marcar MALA ❌ (última foto)")
        btn_train= gr.Button("Entrenar / Actualizar modelo")

    # Cámara backend
    def ui_camera_start():
        cam.start()
        return "Cámara backend encendida"

    def ui_camera_stop():
        cam.stop()
        return "Cámara backend apagada"

    btn_on.click(fn=ui_camera_start, inputs=None, outputs=out_text)
    btn_off.click(fn=ui_camera_stop, inputs=None, outputs=out_text)

    # Botones principales (mantienen su función original)
    btn_snap.click(fn=ui_take_photo_backend, inputs=None, outputs=[out_text, out_img])                 # solo foto
    btn_pred.click(fn=ui_classify_backend, inputs=None, outputs=[out_text, lbl, out_img])             # foto + clasificar
    btn_good.click(fn=ui_save_good_backend, inputs=None, outputs=[out_text, out_img])                 # guarda última
    btn_bad.click(fn=ui_save_bad_backend, inputs=None, outputs=[out_text, out_img])                   # guarda última
    btn_train.click(fn=ui_train, inputs=None, outputs=out_text)                                       # entrena

    # Timer: refresca imagen + resultados si la API externa tomó y predijo una nueva
    gr.Timer(0.5).tick(
        fn=ui_pull_last,
        inputs=[ver_state],
        outputs=[ver_state, out_img, lbl, out_text]
    )

# Montar Gradio dentro de FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")

# ==========================
# Ejecución
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
