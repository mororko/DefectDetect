# DefectDetect

Sistema de detección de defectos con **FastAPI + Gradio + OpenCV**.

## 🚀 Instalación

Clona este repositorio:

```bash
git clone https://github.com/tuusuario/DefectDetect.git
cd DefectDetect
```

Crea un entorno virtual e instala dependencias:

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / Mac
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Uso

Inicia el servidor:

```bash
python app.py
```

Abre en tu navegador:

- **UI de Gradio:** [http://127.0.0.1:7860/ui](http://127.0.0.1:7860/ui)
- **API docs (Swagger):** [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs)

## ⚙️ Variables de entorno de la cámara

`app.py` usa las variables `CAMERA_INDEX`, `CAMERA_WIDTH` y `CAMERA_HEIGHT` 
para configurar la webcam (por defecto 0, 1280 y 720).
Configúralas antes de ejecutar si necesitas otros valores.

```bash
export CAMERA_INDEX=1
export CAMERA_WIDTH=640
export CAMERA_HEIGHT=480
python app.py
```

## 📸 Flujo de trabajo

1. **Encender cámara (backend)** desde la UI o con `POST /camera/start`.
2. **Tomar foto (backend)** → snapshot guardado en la app.
3. **Marcar BUENA/MALA** para crear dataset en `data/good` y `data/bad`.
4. **Entrenar modelo** (`POST /train`) → genera `models/model.pkl`.
5. **Clasificar** (`POST /predict` o botón en UI).

## 🌐 API principal

- `POST /camera/start` → enciende cámara backend.
- `POST /camera/stop` → apaga cámara.
- `GET  /camera/status` → estado de la cámara.
- `POST /predict` → clasifica (archivo o webcam).
- `POST /predict_webcam` → clasifica tomando snapshot.
- `POST /feedback` → guarda archivo etiquetado.
- `POST /feedback_webcam` → guarda snapshot etiquetado.
- `POST /train` → entrena modelo.

## 🔧 Resetear dataset / modelo

Para empezar de cero, borra:

- `models/model.pkl`
- imágenes dentro de `data/good/` y `data/bad/`

Actualmente no se incluye un script `reset_dataset.py`. Si quieres automatizar este proceso, crea un archivo con ese nombre que elimine `models/model.pkl` y limpie las carpetas `data/good/` y `data/bad/`.

## 📦 Requisitos

Ver archivo `requirements.txt`.

## 📜 Licencia

MIT
