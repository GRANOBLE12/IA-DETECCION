"""
gradio_app.py — Interfaz web para HuggingFace Spaces

Sistema de Reconocimiento de Señales de Tránsito
Agente Dual-Transformer: DeiT-tiny (Vision) + Flan-T5 (Lenguaje)

Despliegue: https://huggingface.co/spaces
"""

import os
import re
import time
import threading
import warnings
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    ViTForImageClassification,
    T5ForConditionalGeneration,
    AutoTokenizer,
    logging as hf_logging,
)

# ── Shim: HfFolder fue removido de huggingface_hub>=0.30 pero Gradio <=5.x
# lo importa en oauth.py. Inyectamos un stub antes de que Gradio cargue.
import huggingface_hub as _hfh
if not hasattr(_hfh, "HfFolder"):
    class _HfFolderShim:
        @staticmethod
        def get_token():   return None
        @staticmethod
        def save_token(t): pass
        @staticmethod
        def delete_token():pass
    _hfh.HfFolder = _HfFolderShim

import gradio as gr

# ── Patch: bug en gradio_client 5.0.0 — get_type() falla cuando schema es bool
# gradio_client/utils.py:882  if "const" in schema  → TypeError si schema=True/False
# Parcheamos la funcion para que maneje valores no-dict graciosamente.
try:
    import gradio_client.utils as _gcu
    _orig_get_type = _gcu.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    _gcu.get_type = _safe_get_type
except Exception:
    pass  # si falla el patch, Gradio sigue igual

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

from labels import CLASS_NAMES, DESCRIPTIONS

# ─── Configuracion ────────────────────────────────────────────────────────────

VIT_ARCH    = "facebook/deit-tiny-patch16-224"
VIT_WEIGHTS = "vit_gtsrb.pth"
T5_MODEL    = "google/flan-t5-base"
NUM_CLASSES = 43
IMAGE_SIZE  = 224
CONF_MIN    = 0.75   # Mas permisivo en web (imagenes claras sin ruido de camara)

# Paleta de colores (RGB para PIL)
C_ORANGE  = (255, 165,  0)
C_WHITE   = (255, 255, 255)
C_BLACK   = ( 20,  20,  20)
C_GREEN   = ( 80, 200, 120)

# ─── Utilidades ───────────────────────────────────────────────────────────────

KEYWORDS = {
    "velocidad": [0,1,2,3,4,5,6,7,8],
    "limite":    [0,1,2,3,4,5,6,7,8],
    "20":[0], "30":[1], "50":[2], "60":[3],
    "70":[4], "80":[5], "100":[7], "120":[8],
    "adelantar": [9,10,41,42], "prohibido": [9,10,15,16,17],
    "paso":      [13,15], "entrar": [17], "contravia": [17],
    "prioridad": [11,12], "principal": [12],
    "ceda":      [13], "yield": [13],
    "stop":[14], "alto":[14], "pare":[14], "detener":[14],
    "vehiculo":  [15,16], "camion": [10,16],
    "precaucion":[18], "peligro": [18,19,20,21,22,23,30,31],
    "curva":     [19,20,21], "izquierda":[19,34,37,39],
    "derecha":   [20,33,36,38], "doble":[21],
    "pavimento": [22,23], "deslizante":[23], "hielo":[30],
    "obras":     [25], "semaforo":[26],
    "peatones":  [27], "ninos":[28], "escolar":[28],
    "ciclistas": [29], "bicicleta":[29],
    "animales":  [31], "rotonda":[40], "glorieta":[40],
    "fin":       [6,32,41,42], "giro":[33,34],
    "recto":     [35,36,37],
}

# ─── Carga de modelos ─────────────────────────────────────────────────────────

print("Cargando Transformer 1: DeiT-tiny (Vision)...")
_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_vit_model = ViTForImageClassification.from_pretrained(
    VIT_ARCH, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
)
_vit_model.load_state_dict(
    torch.load(VIT_WEIGHTS, map_location="cpu", weights_only=True)
)
_vit_model.eval()
print("DeiT-tiny listo.")

print("Cargando Transformer 2: Flan-T5-base (Lenguaje)...")
_t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
_t5_model     = T5ForConditionalGeneration.from_pretrained(T5_MODEL)
_t5_model.eval()
print("Flan-T5 listo.\n")


# ─── Funciones del pipeline ───────────────────────────────────────────────────

@torch.no_grad()
def _classify(crop_bgr: np.ndarray):
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = _TRANSFORM(rgb).unsqueeze(0)
    logits = _vit_model(pixel_values=tensor).logits
    probs  = F.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)
    cid = idx.item()
    return cid, CLASS_NAMES.get(cid, f"Clase {cid}"), conf.item()


def _detect(frame_bgr: np.ndarray):
    """Detecta regiones de color (rojo/amarillo/azul) en la imagen."""
    h_img, w_img = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask_r1 = cv2.inRange(hsv, np.array([0,   40, 40]),  np.array([12,  255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([158, 40, 40]),  np.array([180, 255, 255]))
    mask_y  = cv2.inRange(hsv, np.array([15,  60, 70]),  np.array([42,  255, 255]))
    mask_b  = cv2.inRange(hsv, np.array([95,  80, 60]),  np.array([135, 255, 255]))

    mask = mask_r1 | mask_r2 | mask_y | mask_b
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > h_img * w_img * 0.6:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (0.18 <= w / max(h, 1) <= 5.0):
            continue
        m = int(max(w, h) * 0.12)
        boxes.append([
            max(0, x-m), max(0, y-m),
            min(w_img, x+w+m), min(h_img, y+h+m)
        ])

    # NMS simple
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=float)
    order = list(range(len(boxes)))
    keep  = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            b1, b2 = boxes_np[i], boxes_np[j]
            xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
            xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
            a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
            if inter / (a1 + a2 - inter + 1e-6) < 0.3:
                rest.append(j)
        order = rest
    return [boxes[i] for i in keep]


def _t5_explain(class_id: int) -> str:
    nombre = CLASS_NAMES.get(class_id, "Señal desconocida")
    desc   = DESCRIPTIONS.get(class_id, "")
    prompt = (
        f"Question: What does the traffic sign '{nombre}' mean?\n"
        f"Context: {desc}\n"
        f"Answer in Spanish:"
    )
    inputs = _t5_tokenizer(prompt, return_tensors="pt",
                            max_length=200, truncation=True)
    with torch.no_grad():
        out = _t5_model.generate(
            **inputs, max_new_tokens=80,
            num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
        )
    texto = _t5_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    garbled = ["zone de", "seal:", "significado:", "sign name",
               "instructor", "driving", "question:", "context:"]
    if len(texto) > 15 and not any(g in texto.lower() for g in garbled):
        return texto
    return f"{nombre}: {desc}"


def _find_signs(question: str):
    q = question.lower()
    for c in "áéíóúüñ":
        q = q.replace(c, {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u","ñ":"n"}[c])
    scores = {}
    for kw, ids in KEYWORDS.items():
        if kw in q:
            for cid in ids:
                scores[cid] = scores.get(cid, 0) + 1
    for cid, name in CLASS_NAMES.items():
        nm = name.lower().replace("á","a").replace("é","e").replace("í","i")\
                         .replace("ó","o").replace("ú","u")
        hits = sum(1 for w in nm.split() if len(w) > 3 and w in q)
        if hits:
            scores[cid] = scores.get(cid, 0) + hits * 2
    return sorted(scores, key=lambda x: -scores[x])[:3]


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTE REACTIVO PARA WEBCAM EN TIEMPO REAL
# Estado global compartido entre frames (igual que agente.py)
# ═══════════════════════════════════════════════════════════════════════════════

TRIGGER_FRAMES = 3      # frames consecutivos con misma señal para activar T5
CONF_REALTIME  = 0.85   # umbral mas permisivo para webcam (frames con ruido)

_agent_state = {
    "lock":          threading.Lock(),
    "last_class_id": -1,
    "consecutive":   0,
    "explanation":   "🎥 Apunta la cámara a una señal de tránsito...",
    "generating":    False,
    "current_sign":  "",
}


def _t5_async(class_id: int, sign_name: str):
    """Lanza Flan-T5 en thread separado — el stream no se bloquea."""
    with _agent_state["lock"]:
        _agent_state["generating"]  = True
        _agent_state["explanation"] = f"🔄 Analizando '{sign_name}' con Flan-T5..."

    def _run():
        text = _t5_explain(class_id)
        with _agent_state["lock"]:
            _agent_state["explanation"] = (
                f"## 🚦 {sign_name}\n\n"
                f"### 💬 Explicación (Flan-T5):\n{text}\n\n"
                f"### 📋 Descripción técnica:\n{DESCRIPTIONS.get(class_id, '')}"
            )
            _agent_state["generating"] = False

    threading.Thread(target=_run, daemon=True).start()


def procesar_webcam_frame(frame):
    """
    Ciclo del agente para cada frame de la webcam (igual que agente.py):
      1. PERCIBIR  → detector de color
      2. CLASIFICAR → DeiT-tiny
      3. RAZONAR   → señal nueva + estable?
      4. ACTUAR    → lanzar T5 en thread (no bloquea)
    """
    if frame is None:
        with _agent_state["lock"]:
            return None, _agent_state["explanation"]

    # Normalizar a RGB
    if frame.mode != "RGB":
        frame = frame.convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    # ── PERCIBIR + CLASIFICAR ──────────────────────────────────────────────────
    boxes = _detect(frame_bgr)
    best  = None
    if boxes:
        candidates = []
        for box in boxes:
            x1, y1, x2, y2 = box
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cid, name, conf = _classify(crop)
            if conf >= CONF_REALTIME:
                candidates.append((cid, name, conf, box))
        if candidates:
            best = max(candidates, key=lambda x: x[2])

    # ── DIBUJAR sobre el frame ─────────────────────────────────────────────────
    img_out = frame.copy()
    if best:
        cid, name, conf, box = best
        draw = ImageDraw.Draw(img_out)
        x1, y1, x2, y2 = box

        # Caja gruesa
        for o in range(3):
            draw.rectangle([x1-o, y1-o, x2+o, y2+o], outline=C_ORANGE)

        # Etiqueta arriba de la caja
        label = f"{name}  {conf:.0%}"
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        bb     = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        ty     = max(y1 - th - 10, 2)
        draw.rectangle([x1, ty, x1+tw+10, ty+th+6], fill=C_ORANGE)
        draw.text((x1+5, ty+3), label, fill=C_BLACK, font=font)

        # ── RAZONAR: estabilidad temporal antes de activar T5 ──────────────────
        with _agent_state["lock"]:
            if cid == _agent_state["last_class_id"]:
                _agent_state["consecutive"] += 1
            else:
                _agent_state["consecutive"]   = 1
                _agent_state["last_class_id"] = cid

            should_trigger = (
                _agent_state["consecutive"] == TRIGGER_FRAMES
                and not _agent_state["generating"]
                and _agent_state["current_sign"] != name
            )
            if should_trigger:
                _agent_state["current_sign"] = name

        # ── ACTUAR: lanzar Flan-T5 en thread ───────────────────────────────────
        if should_trigger:
            _t5_async(cid, name)

    # Devolver frame anotado + texto actual (se va actualizando solo)
    with _agent_state["lock"]:
        return img_out, _agent_state["explanation"]


# ─── Handlers de Gradio ───────────────────────────────────────────────────────

def procesar_imagen(image):
    """
    Pipeline completo: imagen → deteccion → ViT → Flan-T5 → resultado visual.
    """
    if image is None:
        return None, "⚠️ Sube una imagen para analizar."

    # Normalizar a RGB (Gradio 5 puede entregar RGBA/P si es PNG con alpha)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # PIL → BGR numpy
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detectar y clasificar
    boxes = _detect(frame)
    if not boxes:
        return image, (
            "## ⚠️ No se detectaron señales\n\n"
            "El detector no encontró regiones de color rojo, amarillo o azul.\n\n"
            "**Sugerencias:**\n"
            "- Acerca más la imagen a la señal\n"
            "- Usa una imagen con buena iluminación\n"
            "- Asegúrate de que la señal sea visible y no esté muy lejos"
        )

    results = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cid, name, conf = _classify(crop)
        results.append((cid, name, conf, box))

    if not results:
        return image, "## ⚠️ Señales encontradas pero no clasificadas con certeza."

    # Tomar la de mayor confianza
    best = max(results, key=lambda x: x[2])
    cid, name, conf, box = best

    if conf < CONF_MIN:
        return image, (
            f"## ⚠️ Confianza insuficiente ({conf:.1%})\n\n"
            f"El modelo detectó algo parecido a **{name}** "
            f"pero con baja certeza.\n\n"
            "Intenta con una imagen más clara y frontal de la señal."
        )

    # Dibujar sobre la imagen con PIL (soporta Unicode)
    img_pil = image.copy()
    draw    = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = box

    # Caja principal
    for offset in range(3):
        draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset],
                       outline=C_ORANGE)

    # Etiqueta superior
    label = f"{name}  {conf:.0%}"
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    bbox_t  = draw.textbbox((0, 0), label, font=font)
    tw, th  = bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]
    ty      = max(y1 - th - 10, 2)
    draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=C_ORANGE)
    draw.text((x1 + 5, ty + 3), label, fill=C_BLACK, font=font)

    # Generar explicación con Flan-T5
    explicacion = _t5_explain(cid)
    desc_base   = DESCRIPTIONS.get(cid, "")

    # Icono según tipo de señal
    icono = "🔴" if cid <= 17 else ("⚠️" if cid <= 31 else "🔵")

    resultado_md = f"""
## {icono} {name}

---

### 🤖 Transformer 1 — DeiT-tiny (Vision)
| | |
|---|---|
| **Señal detectada** | {name} |
| **Confianza** | **{conf:.1%}** |
| **Clase GTSRB** | #{cid} |

---

### 💬 Transformer 2 — Flan-T5 (Lenguaje)
> *Explicación generada automáticamente por el modelo de lenguaje:*

{explicacion}

---

### 📋 Descripción técnica
{desc_base}
"""
    return img_pil, resultado_md


def responder_pregunta(pregunta, historial):
    """
    Asistente Q&A: palabras clave → Flan-T5 → respuesta en español.
    """
    if not pregunta.strip():
        return historial, ""

    ids = _find_signs(pregunta)
    if not ids:
        respuesta = (
            "No encontré una señal específica para esa pregunta.\n"
            "Prueba mencionando: stop, ceda el paso, velocidad, curva, "
            "peatones, obras, semáforo, etc."
        )
    else:
        cid    = ids[0]
        nombre = CLASS_NAMES[cid]
        desc   = DESCRIPTIONS[cid]

        prompt = (
            f"Question: {pregunta}\n"
            f"Context about traffic sign '{nombre}': {desc}\n"
            f"Answer in Spanish:"
        )
        inputs = _t5_tokenizer(prompt, return_tensors="pt",
                                max_length=250, truncation=True)
        with torch.no_grad():
            out = _t5_model.generate(
                **inputs, max_new_tokens=100,
                num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
            )
        t5_out = _t5_tokenizer.decode(out[0], skip_special_tokens=True).strip()

        garbled = ["zone de", "seal:", "sign name", "question:", "context:"]
        if len(t5_out) > 15 and not any(g in t5_out.lower() for g in garbled):
            respuesta = t5_out
        else:
            respuesta = f"{nombre}: {desc}"

    historial.append((pregunta, respuesta))
    return historial, ""


# ─── Interfaz Gradio ──────────────────────────────────────────────────────────

CSS = """
/* ── Fondo general ── */
body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Header degradado ── */
.header-box {
    background: linear-gradient(135deg, #1a0a00 0%, #2d1600 40%, #0a0a1a 100%);
    border: 1px solid #ff8c00;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 8px;
    text-align: center;
}
.header-box h1 {
    color: #ff9500;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.header-box p {
    color: #aaa;
    font-size: 0.95rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #ff8c00;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    color: #ff9500;
    margin: 6px 4px 0;
}

/* ── Tabs ── */
.tab-nav button {
    background: #111 !important;
    color: #aaa !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 24px !important;
    border: 1px solid #333 !important;
    border-bottom: none !important;
}
.tab-nav button.selected {
    background: #1a1a2e !important;
    color: #ff9500 !important;
    border-color: #ff8c00 !important;
}

/* ── Paneles ── */
.panel-dark {
    background: #111118;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 20px;
}

/* ── Botón principal ── */
.btn-primary {
    background: linear-gradient(90deg, #ff6b00, #ff9500) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 1rem !important;
    padding: 12px 0 !important;
    transition: all 0.2s !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(255,140,0,0.4) !important;
}

/* ── Input de imagen ── */
.image-upload {
    border: 2px dashed #333 !important;
    border-radius: 12px !important;
    background: #0d0d18 !important;
    min-height: 280px !important;
}
.image-upload:hover {
    border-color: #ff8c00 !important;
}

/* ── Output de texto ── */
.output-md {
    background: #0d0d18 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 12px !important;
    padding: 16px !important;
    min-height: 280px !important;
    color: #ddd !important;
}

/* ── Chatbot ── */
.chatbot {
    background: #0d0d18 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 12px !important;
}
.chatbot .message.user {
    background: #1a1a2e !important;
    color: #fff !important;
    border-radius: 10px 10px 4px 10px !important;
}
.chatbot .message.bot {
    background: #1e120a !important;
    color: #eee !important;
    border-left: 3px solid #ff8c00 !important;
    border-radius: 4px 10px 10px 10px !important;
}

/* ── Texto de info ── */
.info-card {
    background: #111118;
    border: 1px solid #2a2a3a;
    border-left: 3px solid #ff8c00;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 8px;
    color: #999;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    color: #555;
    font-size: 0.78rem;
    margin-top: 16px;
    padding: 12px;
    border-top: 1px solid #1a1a2e;
}
"""

EJEMPLOS_IMAGENES = []  # Se pueden agregar si hay imagenes de ejemplo en el repo

with gr.Blocks(css=CSS, title="Reconocimiento de Señales de Tránsito") as demo:

    # ── Header ──
    gr.HTML("""
    <div class="header-box">
        <h1>🚦 Reconocimiento de Señales de Tránsito</h1>
        <p>Sistema de visión por computador con agente dual-transformer</p>
        <span class="badge">🔭 DeiT-tiny (Vision Transformer)</span>
        <span class="badge">💬 Flan-T5-base (Text Transformer)</span>
        <span class="badge">📊 99.05% precisión en GTSRB</span>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════
        # TAB 1 — TIEMPO REAL (WEBCAM STREAMING)
        # Mismo pipeline que agente.py local
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("🎥 Tiempo Real"):
            gr.HTML("""
            <div class="info-card">
                <b>Agente en tiempo real con webcam:</b><br>
                🎯 Detección continua frame por frame (igual que <code>agente.py</code> local)<br>
                🤖 <b>DeiT-tiny</b> clasifica cada frame; <b>Flan-T5</b> se activa
                cuando una señal aparece de forma <b>estable</b> (3 frames seguidos)<br>
                🔒 Tu navegador pedirá <b>permiso de cámara</b> — acéptalo.
                El video <b>no se envía a ningún servidor</b> excepto al de HF Spaces.
            </div>
            """)

            with gr.Row():
                webcam_in = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="pil",
                    label="📷 Cámara en vivo",
                    height=380,
                    mirror_webcam=True,
                )
                webcam_out = gr.Image(
                    type="pil",
                    label="🎯 Detección del agente",
                    height=380,
                    interactive=False,
                )

            stream_status = gr.Markdown(
                value="🎥 *Espera unos segundos a que cargue la cámara, luego apunta a una señal...*",
                elem_classes=["output-md"],
            )

            # Streaming: cada frame se procesa automaticamente
            webcam_in.stream(
                fn=procesar_webcam_frame,
                inputs=[webcam_in],
                outputs=[webcam_out, stream_status],
                stream_every=0.4,      # ~2.5 fps (suficiente y evita saturar HF)
                show_progress="hidden",
            )

            gr.HTML("""
            <div class="info-card">
                ⚡ <b>Latencia esperada:</b> 300-600ms por frame en HF free tier (sin GPU).<br>
                Si quieres velocidad real (~30 fps) ejecuta localmente:
                <code>python agente.py</code>
            </div>
            """)

        # ══════════════════════════════════════════════════════════
        # TAB 2 — AGENTE VISUAL (imagen estática)
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("🤖 Agente Visual"):
            gr.HTML("""
            <div class="info-card">
                <b>Cómo funciona el agente:</b><br>
                1️⃣ <b>Transformer 1 — DeiT-tiny</b>: detecta y clasifica la señal en la imagen (99.05% precisión)<br>
                2️⃣ <b>Transformer 2 — Flan-T5</b>: genera automáticamente una explicación en español<br>
                Sube una foto de cualquier señal de tránsito alemana o colombiana.
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="pil",
                        label="📷 Imagen de señal de tránsito",
                        elem_classes=["image-upload"],
                        sources=["upload", "webcam", "clipboard"],
                        height=320,
                    )
                    btn_analizar = gr.Button(
                        "🔍  Analizar señal",
                        variant="primary",
                        elem_classes=["btn-primary"],
                    )

                with gr.Column(scale=1):
                    img_output = gr.Image(
                        type="pil",
                        label="🎯 Señal detectada",
                        height=320,
                        interactive=False,
                    )

            resultado_md = gr.Markdown(
                value="*Los resultados del agente aparecerán aquí...*",
                elem_classes=["output-md"],
            )

            btn_analizar.click(
                fn=procesar_imagen,
                inputs=[img_input],
                outputs=[img_output, resultado_md],
            )

            gr.HTML("""
            <div class="info-card">
                🇩🇪 Compatible con señales alemanas (GTSRB) &nbsp;|&nbsp;
                🇨🇴 Compatible con señales colombianas &nbsp;|&nbsp;
                43 categorías de señales
            </div>
            """)

        # ══════════════════════════════════════════════════════════
        # TAB 2 — ASISTENTE Q&A
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("💬 Asistente"):
            gr.HTML("""
            <div class="info-card">
                <b>Asistente conversacional</b> — Recuperación por palabras clave + Flan-T5-base<br>
                Pregunta lo que quieras sobre señales de tránsito en español.
            </div>
            """)

            chatbot = gr.Chatbot(
                label="Conversación",
                height=380,
                elem_classes=["chatbot"],
                type="tuples",
            )

            with gr.Row():
                txt_pregunta = gr.Textbox(
                    placeholder="Ej: ¿Qué significa la señal de ceda el paso?",
                    label="",
                    scale=5,
                    container=False,
                )
                btn_enviar = gr.Button(
                    "Enviar ➤",
                    variant="primary",
                    scale=1,
                    elem_classes=["btn-primary"],
                    min_width=120,
                )

            gr.Examples(
                examples=[
                    ["¿Qué significa ceda el paso?"],
                    ["¿Qué hago si hay una señal de STOP?"],
                    ["¿Qué significa el límite de velocidad de 50?"],
                    ["¿Qué indica una curva peligrosa a la derecha?"],
                    ["¿Hay señales para peatones?"],
                    ["¿Qué significa fin de prohibición de adelantar?"],
                ],
                inputs=txt_pregunta,
                label="Ejemplos de preguntas",
            )

            btn_enviar.click(
                fn=responder_pregunta,
                inputs=[txt_pregunta, chatbot],
                outputs=[chatbot, txt_pregunta],
            )
            txt_pregunta.submit(
                fn=responder_pregunta,
                inputs=[txt_pregunta, chatbot],
                outputs=[chatbot, txt_pregunta],
            )

        # ══════════════════════════════════════════════════════════
        # TAB 3 — ACERCA DEL PROYECTO
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("📋 Acerca del proyecto"):
            gr.Markdown("""
## 🚦 Sistema de Reconocimiento de Señales de Tránsito

### Arquitectura dual-transformer

Este sistema implementa un **agente reactivo** que conecta dos modelos transformer:

| Componente | Modelo | Parámetros | Función |
|---|---|---|---|
| **Transformer 1** | DeiT-tiny (Vision ViT) | 5.7M | Clasifica la señal en la imagen |
| **Transformer 2** | Flan-T5-base | 250M | Genera explicación en lenguaje natural |

### Pipeline del agente

```
Imagen de entrada
      ↓
Detector de color HSV (rojo, amarillo, azul)
      ↓
[Transformer 1] DeiT-tiny → "STOP, 99.7% confianza"
      ↓
[Transformer 2] Flan-T5-base → "Detención obligatoria. Para completamente..."
      ↓
Resultado visual + explicación automática
```

### Dataset: GTSRB

El modelo fue entrenado en el **German Traffic Sign Recognition Benchmark**:
- 43 categorías de señales de tránsito
- ~39,209 imágenes de entrenamiento
- ~12,630 imágenes de prueba
- **Precisión obtenida: 99.05%** en el conjunto de prueba

### Técnicas utilizadas

- **Transfer Learning**: DeiT-tiny pre-entrenado en ImageNet-1k → fine-tuning en GTSRB
- **Beam Search**: generación de texto con 4 hipótesis paralelas en Flan-T5
- **Detección por color**: máscaras HSV para localizar señales sin modelos adicionales
- **RAG simplificado**: recuperación por palabras clave + generación con T5

### Entrenamiento

| Época | Train | Validación | Tiempo |
|---|---|---|---|
| 1 | 91.2% | 98.6% | ~25 min |
| 2 | 99.6% | 99.8% | ~25 min |
| **TEST** | — | **99.05%** | — |

---
*Andres Felipe Bolaños Zuñiga — Inteligencia Artificial, Tercer Corte — 2026*
            """)

    # ── Footer ──
    gr.HTML("""
    <div class="footer-text">
        🚦 Sistema de Reconocimiento de Señales de Tránsito &nbsp;·&nbsp;
        DeiT-tiny + Flan-T5-base &nbsp;·&nbsp;
        Andres Felipe Bolaños Zuñiga &nbsp;·&nbsp; 2026
    </div>
    """)


# En HF Spaces el 'spaces' package intercepta demo.launch().
# show_api=False evita el bug de Gradio 5.0.0 en get_api_info():
#   "TypeError: argument of type 'bool' is not iterable"
# que causa el "No API found" en el UI.
demo.queue()
demo.launch(show_api=False)
