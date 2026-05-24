# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comandos principales

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo DeiT-tiny en GTSRB (requerido antes de ejecutar inferencia)
python train_vit.py                   # 2 épocas (~50 min en CPU), guarda vit_gtsrb.pth
python train_vit.py --epochs 3        # más épocas
python train_vit.py --no-cache        # reentrenar aunque exista el modelo

# App de escritorio con OpenCV (solo clasificador ViT, sin Flan-T5)
python app.py                         # webcam
python app.py --image foto.jpg        # imagen estática
python app.py --video ruta.mp4        # archivo de video
python app.py --no-detector           # sin ColorDetector, clasifica centro del frame

# Agente reactivo dual-transformer (DeiT-tiny + Flan-T5)
python agente.py                      # webcam
python agente.py --image foto.jpg
python agente.py --video ruta.mp4

# Asistente de texto en terminal (solo Flan-T5 + búsqueda por palabras clave)
python assistant.py

# Interfaz web Gradio (HuggingFace Spaces)
python gradio_app.py
```

## Arquitectura

El sistema implementa un **agente reactivo dual-transformer** para reconocimiento de señales GTSRB (43 clases):

```
Frame/imagen
    ↓
TrafficSignDetector (detector.py)
    ├── ColorDetector: máscaras HSV (rojo, amarillo, azul) → bounding boxes
    └── YoloDetector: YOLOv8n COCO (modo video/webcam, complemento)
    ↓
[Transformer 1] DeiT-tiny (facebook/deit-tiny-patch16-224)
    Pesos: vit_gtsrb.pth — fine-tuning en GTSRB, 99.05% precisión
    Entrada: crop BGR 224×224 normalizado con ImageNet stats
    Salida: class_id (0-42) + confianza
    ↓
[Transformer 2] Flan-T5-base (google/flan-t5-base)
    Entrada: prompt Q&A con nombre + descripción de la señal
    Salida: explicación en español (beam search, 4 haces)
```

### Módulos

| Archivo | Rol |
|---|---|
| `train_vit.py` | Fine-tuning de DeiT-tiny en GTSRB; genera `vit_gtsrb.pth` |
| `detector.py` | `ColorDetector` (HSV), `YoloDetector` (YOLOv8), `TrafficSignDetector` (selector) |
| `labels.py` | `CLASS_NAMES` y `DESCRIPTIONS` — diccionarios para las 43 clases GTSRB |
| `app.py` | App OpenCV: solo ViT, sin T5, panel lateral con descripciones |
| `agente.py` | Agente completo: ViT + T5 en hilo separado, panel reactivo |
| `assistant.py` | Chatbot terminal: búsqueda por keywords + Flan-T5 |
| `gradio_app.py` | Interfaz web: streaming webcam + imagen estática + Q&A |

### Decisiones de diseño clave

- **`vit_gtsrb.pth` debe existir** antes de ejecutar `app.py`, `agente.py` o `gradio_app.py`. Si no existe, el programa termina con error indicando que hay que correr `train_vit.py`.
- **Flan-T5 se lanza en un hilo separado** (`agente.py`, `gradio_app.py`) para no bloquear el stream de video mientras genera la explicación (~3 s en CPU).
- **El agente requiere `TRIGGER_FRAMES` consecutivos** de la misma señal antes de activar Flan-T5, para evitar falsos positivos.
- **OpenCV `putText` no soporta UTF-8** — `agente.py` usa `_limpiar()` para reemplazar tildes/ñ antes de dibujar texto en pantalla. `gradio_app.py` usa PIL/Pillow que sí admite Unicode.
- **`gradio_app.py` incluye parches de compatibilidad** para Gradio 5.x + Python 3.13: shim `HfFolder` (removido de `huggingface_hub>=0.30`) y patch `get_type()` en `gradio_client`.
- El `ColorDetector` usa NMS propio (sin dependencias extra) para eliminar cajas solapadas.

## Despliegue

Push a la rama `main` activa el workflow `.github/workflows/deploy_hf.yml`, que hace force-push al Space `GRANOBLE12/reconocimiento-senales-transito` en HuggingFace. El secreto `HF_TOKEN` debe estar configurado en el repositorio de GitHub. El Space usa `hf_requirements.txt` (no `requirements.txt`).
