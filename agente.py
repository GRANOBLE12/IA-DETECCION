"""
agente.py — Agente reactivo que conecta Vision Transformer + Text Transformer.

Arquitectura del agente (sistema dual-transformer):

  PERCEPCION   ->  Frame de video (webcam / archivo / imagen)
  DETECCION    ->  ColorDetector detecta regiones candidatas a senales
  CLASIFICACION->  [Transformer 1] DeiT-tiny identifica QUE senal es
  RAZONAMIENTO ->  Agente evalua: es nueva? hay confianza suficiente?
  ACCION       ->  [Transformer 2] Flan-T5 genera explicacion automatica

A diferencia de app.py (solo etiqueta visual) y assistant.py (solo texto),
este agente CONECTA los dos transformers en un pipeline autonomo:

    imagen → DeiT-tiny → "STOP (99%)"  →  Flan-T5  → "Detencion obligatoria..."

El agente actua de forma autonoma al detectar una senal nueva.
No requiere preguntas del usuario — es REACTIVO a la imagen.

Uso:
    python agente.py                       # Webcam
    python agente.py --video archivo.mp4   # Archivo de video
    python agente.py --image foto.jpg      # Imagen estatica (explica cada senal)
"""

import os
import sys
import time
import argparse
import threading
import textwrap

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import (
    ViTForImageClassification,
    T5ForConditionalGeneration,
    AutoTokenizer,
    logging as hf_logging,
)
import warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

from labels import CLASS_NAMES, DESCRIPTIONS
from detector import TrafficSignDetector


# ─── Configuracion ───────────────────────────────────────────────────────────

VIT_ARCH       = "facebook/deit-tiny-patch16-224"
VIT_WEIGHTS    = "vit_gtsrb.pth"
T5_MODEL       = "google/flan-t5-base"
NUM_CLASSES    = 43
IMAGE_SIZE     = 224

CONF_MIN       = 0.97   # Confianza minima del ViT para que el agente actue
                        # (0.97 = 97% — solo senales con certeza casi absoluta)
TRIGGER_FRAMES = 6      # Frames consecutivos con misma senal antes de explicar
MIN_AREA_AGENT = 600    # Area minima de region en px^2 para el agente
                        # (mas alto que app.py para reducir ruido en webcam)
PANEL_W        = 400    # Ancho del panel lateral informativo (pixeles)

# Paleta de colores BGR
C_ACCENT = (0, 200, 255)    # Naranja/cian — resaltados
C_GREEN  = (80, 220, 120)   # Verde — T5 listo
C_BLUE   = (255, 160, 60)   # Azul claro — T5 generando
C_WHITE  = (255, 255, 255)
C_GRAY   = (180, 180, 180)
C_DIM    = (110, 110, 110)


# ─── Utilidad: normalizar texto para OpenCV ──────────────────────────────────
# OpenCV putText no soporta UTF-8 — las tildes y la ñ salen como "??"
# Esta funcion las reemplaza por equivalentes ASCII antes de dibujar.

def _limpiar(texto: str) -> str:
    """Reemplaza caracteres no-ASCII para compatibilidad con OpenCV."""
    tabla = str.maketrans(
        "áéíóúÁÉÍÓÚàèìòùäëïöüñÑüÜ¿¡",
        "aeiouAEIOUaeiouaeiounNuU  "
    )
    return texto.translate(tabla)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER 1 — Vision Transformer (DeiT-tiny)
# Tarea: imagen RGB → clase de senal (0-42) + confianza
# ═══════════════════════════════════════════════════════════════════════════════

class ViTClassifier:
    """
    DeiT-tiny fine-tuneado en GTSRB (99.05% de precision en test).

    Arquitectura Vision Transformer:
      - Divide la imagen 224x224 en 196 parches de 16x16
      - Cada parche se convierte en un vector (embedding)
      - Self-attention compara todos los parches entre si
      - La cabeza de clasificacion produce logits para 43 clases
    """

    TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self):
        if not os.path.exists(VIT_WEIGHTS):
            print(f"\nERROR: No se encontro '{VIT_WEIGHTS}'.")
            print("Primero ejecuta:  python train_vit.py\n")
            sys.exit(1)

        print(f"[Transformer 1 — Vision]  Cargando DeiT-tiny desde '{VIT_WEIGHTS}'...")
        self.model = ViTForImageClassification.from_pretrained(
            VIT_ARCH, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
        )
        self.model.load_state_dict(
            torch.load(VIT_WEIGHTS, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        print(f"[Transformer 1 — Vision]  Listo. (99.05% precision en GTSRB)\n")

    @torch.no_grad()
    def classify(self, crop_bgr: np.ndarray) -> tuple[int, str, float]:
        """
        Clasifica un recorte de imagen BGR.

        Retorna: (class_id, class_name, confianza)
        """
        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.TRANSFORM(rgb).unsqueeze(0)           # (1, 3, 224, 224)
        logits = self.model(pixel_values=tensor).logits     # (1, 43)
        probs  = F.softmax(logits, dim=1)                   # probabilidades
        conf, idx = probs.max(dim=1)
        cid = idx.item()
        return cid, CLASS_NAMES.get(cid, f"Clase {cid}"), conf.item()


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER 2 — Text Transformer (Flan-T5-base)
# Tarea: class_id → explicacion en lenguaje natural
# ═══════════════════════════════════════════════════════════════════════════════

class T5Explainer:
    """
    Flan-T5-base: encoder-decoder Text-to-Text Transformer.

    Recibe la SALIDA del ViT (class_id) y genera automaticamente
    una explicacion en espanol — sin que el usuario pregunte.

    Flujo interno:
      class_id
        -> CLASS_NAMES[cid] + DESCRIPTIONS[cid]   (base de conocimiento)
        -> prompt estructurado                     (instruccion para T5)
        -> Flan-T5 genera tokens                   (beam search, 4 haces)
        -> texto decodificado                      (respuesta en espanol)
    """

    def __init__(self):
        print(f"[Transformer 2 — Lenguaje] Cargando Flan-T5-base...")
        self.tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
        self.model     = T5ForConditionalGeneration.from_pretrained(T5_MODEL)
        self.model.eval()
        print(f"[Transformer 2 — Lenguaje] Listo.\n")

    def explain(self, class_id: int) -> str:
        """
        Genera una explicacion automatica de la senal detectada por el ViT.

        Args:
            class_id: Clase (0-42) identificada por el Vision Transformer.

        Returns:
            Texto en espanol explicando la senal y su implicacion para el conductor.
        """
        nombre = CLASS_NAMES.get(class_id, "Senal desconocida")
        desc   = DESCRIPTIONS.get(class_id, "")

        # Prompt formato Q&A — evita que T5 repita la instruccion
        prompt = (
            f"Question: What does the traffic sign '{nombre}' mean?\n"
            f"Context: {desc}\n"
            f"Answer in Spanish:"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=200, truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,           # Beam search: 4 hipotesis en paralelo
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        t5_out = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Validar que T5 no repitio el prompt ni genero texto corrupto
        garbled = [
            "zone de", "seal:", "significado:", "senal:", "sign name",
            "instructor", "driving", "traffic sign", "what does",
            "answer in", "question:", "context:",
        ]
        tiene_garbling = any(g in t5_out.lower() for g in garbled)

        if len(t5_out) > 15 and not tiene_garbling:
            return _limpiar(t5_out)
        return _limpiar(f"{nombre}: {desc}")    # Fallback — siempre correcto


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTE REACTIVO
# Conecta los dos transformers en un ciclo percibir → razonar → actuar
# ═══════════════════════════════════════════════════════════════════════════════

class TrafficAgent:
    """
    Agente reactivo que conecta el Vision Transformer con el Text Transformer.

    Ciclo del agente (ejecutado en cada frame):
      1. PERCIBIR   → detectar regiones de senales en el frame actual
      2. CLASIFICAR → DeiT-tiny identifica cada senal (Transformer 1)
      3. RAZONAR    → es una senal nueva? hay confianza suficiente?
                      lleva suficientes frames consecutivos?
      4. ACTUAR     → lanzar Flan-T5 en hilo separado (Transformer 2)
                      mostrar explicacion en pantalla cuando este lista

    El agente usa un hilo de fondo para Flan-T5 (~3 seg en CPU)
    de modo que el video no se congela mientras genera la respuesta.
    """

    def __init__(self, detector, vit: ViTClassifier, t5: T5Explainer):
        self.detector = detector
        self.vit      = vit
        self.t5       = t5

        # ── Estado interno del agente ─────────────────────────────────────────
        self._lock          = threading.Lock()
        self.last_class_id  = -1        # Ultima senal que activo el agente
        self.consecutive    = 0         # Frames consecutivos con la misma senal
        self.no_sign_frames = 0         # Frames sin ninguna senal

        # ── Estado compartido con el hilo de visualizacion ────────────────────
        self.sign_name   = ""                           # Nombre de la senal actual
        self.sign_conf   = 0.0                          # Confianza del ViT
        self.explanation = "Apunta la camara a una senal de transito..."
        self.generating  = False                        # T5 esta procesando?
        self.t5_time     = 0.0                          # Segundos que tardo T5

    # ── Paso 1+2: Percibir y Clasificar ──────────────────────────────────────

    def perceive(self, frame: np.ndarray) -> list:
        """
        Detecta regiones candidatas, las clasifica con DeiT-tiny
        y retorna SOLO la mejor deteccion por clase (sin duplicados).

        Retorna: lista de (class_id, class_name, confianza, bounding_box)
                 ordenada por confianza descendente, maximo 1 resultado.
        """
        dets = self.detector.detect(frame)

        # Clasificar cada region detectada
        classified = []
        for det in dets:
            crop = self.detector.crop(frame, det["box"])
            if crop.size == 0:
                continue
            cid, name, conf = self.vit.classify(crop)
            classified.append((cid, name, conf, det["box"]))

        if not classified:
            return []

        # Quedarse solo con la deteccion de mayor confianza
        # (evita multiples cajas superpuestas de la misma o distintas senales)
        best = max(classified, key=lambda x: x[2])

        # Solo retornar si supera el umbral minimo
        if best[2] >= CONF_MIN:
            return [best]
        return []

    # ── Paso 3+4: Razonar y Actuar ────────────────────────────────────────────

    def reason_and_act(self, detections: list):
        """
        Logica de decision del agente:

          - Filtra por confianza minima (CONF_MIN)
          - Requiere TRIGGER_FRAMES consecutivos para evitar falsos positivos
          - Solo actua cuando la senal es diferente a la ultima explicada
          - Lanza Flan-T5 en hilo separado para no bloquear el video
        """
        # Sin detecciones: resetear contador
        if not detections:
            self.no_sign_frames += 1
            if self.no_sign_frames > 30:   # ~1 segundo sin senales
                with self._lock:
                    self.sign_name   = ""
                    self.sign_conf   = 0.0
                    if not self.generating:
                        self.explanation = "Apunta la camara a una senal de transito..."
                self.consecutive = 0
            return

        self.no_sign_frames = 0

        # perceive() ya garantiza que detections[0] es la mejor y supera CONF_MIN
        cid, name, conf, _ = detections[0]

        # Actualizar lo que se muestra en pantalla (siempre, desde frame 1)
        with self._lock:
            self.sign_name = name
            self.sign_conf = conf

        # Contar frames consecutivos de la misma senal
        if cid == self.last_class_id:
            self.consecutive += 1
        else:
            self.consecutive    = 1
            self.last_class_id  = cid

        # ACCION: activar Flan-T5 exactamente cuando se alcanza el umbral
        if self.consecutive == TRIGGER_FRAMES and not self.generating:
            self._act(cid, name)

    def _act(self, class_id: int, sign_name: str):
        """
        ACCION del agente: lanza Flan-T5 en un hilo de fondo.

        El hilo escribe en self.explanation cuando termina.
        El hilo principal sigue leyendo frames sin interrupciones.
        """
        self.generating = True
        with self._lock:
            self.explanation = f"Analizando '{sign_name}'... (Flan-T5 generando)"

        print(f"\n[AGENTE] Senal detectada: {sign_name}")
        print(f"[AGENTE] Activando Flan-T5 para generar explicacion...")

        t_start = time.time()

        def _run_t5():
            text = self.t5.explain(class_id)
            elapsed = time.time() - t_start
            with self._lock:
                self.explanation = text
                self.t5_time     = elapsed
            self.generating = False
            print(f"[AGENTE] Explicacion generada en {elapsed:.1f}s: {text[:80]}...")

        threading.Thread(target=_run_t5, daemon=True).start()

    # ── Visualizacion ─────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Dibuja sobre el frame SOLO la mejor deteccion (una sola caja)
        y el panel lateral con la informacion del agente.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Solo dibujar la primera deteccion (la de mayor confianza)
        # perceive() ya garantiza que hay como maximo 1 elemento en la lista
        if detections:
            cid, name, conf, box = detections[0]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_ACCENT, 2)
            label = f"{name}  {conf:.0%}"
            (tw, th), base = cv2.getTextSize(label, font, 0.52, 1)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, ty-th-4), (x1+tw+6, ty+base), C_ACCENT, -1)
            cv2.putText(frame, label, (x1+2, ty-2), font, 0.52, (20, 20, 20), 1, cv2.LINE_AA)

        return self._build_panel(frame)

    def _build_panel(self, frame: np.ndarray) -> np.ndarray:
        """Construye el panel lateral informativo."""
        h     = frame.shape[0]
        panel = np.full((h, PANEL_W, 3), 15, dtype=np.uint8)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        y     = 24

        # ── Encabezado ──
        cv2.putText(panel, "AGENTE REACTIVO", (10, y), font, 0.52, C_ACCENT, 1, cv2.LINE_AA)
        y += 4
        cv2.line(panel, (10, y), (PANEL_W-10, y), (55, 55, 55), 1)
        y += 20

        # ── Transformer 1: ViT ──
        cv2.putText(panel, "TRANSFORMER 1 — DeiT-tiny (Vision)", (10, y),
                    font, 0.36, C_DIM, 1, cv2.LINE_AA)
        y += 18

        with self._lock:
            sname = self.sign_name
            sconf = self.sign_conf
            expl  = self.explanation
            gen   = self.generating
            t5t   = self.t5_time

        if sname:
            # Nombre de la senal — limpiar tildes para OpenCV
            label = _limpiar(sname if len(sname) <= 32 else sname[:30] + "...")
            cv2.putText(panel, label, (10, y), font, 0.48, C_ACCENT, 1, cv2.LINE_AA)
            y += 20
            cv2.putText(panel, f"Confianza: {sconf:.1%}", (10, y),
                        font, 0.38, C_GRAY, 1, cv2.LINE_AA)
            y += 22

            # Barra de confianza visual
            bw = int((PANEL_W - 20) * sconf)
            cv2.rectangle(panel, (10, y), (PANEL_W-10, y+7), (40, 40, 40), -1)
            color_bar = C_GREEN if sconf > 0.8 else C_ACCENT
            cv2.rectangle(panel, (10, y), (10 + bw, y+7), color_bar, -1)
            y += 18
        else:
            cv2.putText(panel, "Sin senales detectadas", (10, y),
                        font, 0.4, C_DIM, 1, cv2.LINE_AA)
            y += 22

        y += 6
        cv2.line(panel, (10, y), (PANEL_W-10, y), (55, 55, 55), 1)
        y += 16

        # ── Transformer 2: Flan-T5 ──
        if gen:
            t2_label = "TRANSFORMER 2 — Flan-T5  [generando...]"
            t2_color = C_BLUE
        else:
            t2s = f"{t5t:.1f}s" if t5t > 0 else "en espera"
            t2_label = f"TRANSFORMER 2 — Flan-T5  [{t2s}]"
            t2_color = C_GREEN

        cv2.putText(panel, t2_label, (10, y), font, 0.34, t2_color, 1, cv2.LINE_AA)
        y += 18
        cv2.putText(panel, "Explicacion automatica:", (10, y),
                    font, 0.38, C_GRAY, 1, cv2.LINE_AA)
        y += 18

        # Texto envuelto — limpiar tildes para OpenCV antes de dibujar
        chars   = max(1, (PANEL_W - 22) // 7)
        expl_ok = _limpiar(expl)
        for paragraph in expl_ok.split('\n'):
            for line in textwrap.wrap(paragraph or " ", width=chars):
                if y > h - 24:
                    break
                cv2.putText(panel, line, (12, y), font, 0.37, C_WHITE, 1, cv2.LINE_AA)
                y += 15

        # Barra de progreso animada mientras T5 genera
        if gen:
            bar_y = h - 14
            t_anim = int(time.time() * 4) % (PANEL_W - 20)
            cv2.rectangle(panel, (10, bar_y-5), (PANEL_W-10, bar_y+5), (35,35,35), -1)
            seg_end = min(10 + t_anim + 80, PANEL_W - 10)
            cv2.rectangle(panel, (10 + t_anim, bar_y-4), (seg_end, bar_y+4), C_BLUE, -1)
            cv2.putText(panel, "Flan-T5 procesando...", (10, bar_y-10),
                        font, 0.32, C_DIM, 1, cv2.LINE_AA)

        # Numero de frames y contador
        y = h - 8
        frames_txt = f"frames: {self.consecutive}/{TRIGGER_FRAMES}  |  umbral conf: {CONF_MIN:.0%}"
        cv2.putText(panel, frames_txt, (10, y), font, 0.30, C_DIM, 1, cv2.LINE_AA)

        return np.hstack([frame, panel])


# ─── Modo imagen estatica ────────────────────────────────────────────────────

def run_image(path: str, agent: TrafficAgent):
    """Procesa una imagen: detecta todas las senales y explica cada una."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"Error: no se pudo abrir '{path}'")
        sys.exit(1)

    print(f"\nProcesando imagen: {path}")
    detections = agent.perceive(frame)
    print(f"Senales detectadas: {len([d for d in detections if d[2] >= CONF_MIN])}")

    if not detections:
        print("No se detectaron senales. Intenta con --video o webcam.")
    else:
        # Explicar cada senal secuencialmente (sin threading en modo imagen)
        best = max(detections, key=lambda x: x[2])
        for cid, name, conf, box in sorted(detections, key=lambda x: -x[2]):
            if conf < CONF_MIN:
                continue
            print(f"\n  Senal: {name}  ({conf:.0%})")
            print(f"  Generando explicacion con Flan-T5...")
            t0   = time.time()
            expl = agent.t5.explain(cid)
            print(f"  Explicacion ({time.time()-t0:.1f}s): {expl}")

            with agent._lock:
                agent.sign_name   = name
                agent.sign_conf   = conf
                agent.explanation = expl
                agent.t5_time     = time.time() - t0

    display = agent.draw(frame, detections)
    h, w = display.shape[:2]
    if h > 900 or w > 1500:
        sc = min(900/h, 1500/w)
        display = cv2.resize(display, (int(w*sc), int(h*sc)))

    cv2.namedWindow("Agente de Senales de Transito", cv2.WINDOW_NORMAL)
    cv2.imshow("Agente de Senales de Transito", display)
    print("\nPresiona cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── Modo video / webcam ─────────────────────────────────────────────────────

def run_video(source, agent: TrafficAgent):
    """Bucle principal para video en tiempo real."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        msg = "la webcam" if source == 0 else f"'{source}'"
        print(f"Error: no se pudo abrir {msg}")
        sys.exit(1)

    fuente = "Webcam" if source == 0 else source
    print(f"Fuente de video: {fuente}")
    print("Presiona 'q' para salir.\n")

    cv2.namedWindow("Agente de Senales de Transito", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ciclo completo del agente
        detections = agent.perceive(frame)         # PERCIBIR + CLASIFICAR
        agent.reason_and_act(detections)            # RAZONAR + ACTUAR
        display = agent.draw(frame, detections)     # VISUALIZAR

        cv2.imshow("Agente de Senales de Transito", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Agente reactivo: DeiT-tiny (Vision Transformer) + Flan-T5 (Text Transformer)"
    )
    p.add_argument("--image", type=str, help="Ruta a una imagen con senales")
    p.add_argument("--video", type=str, help="Ruta a un archivo de video")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 62)
    print("  AGENTE REACTIVO DE SENALES DE TRANSITO")
    print("  Pipeline dual-transformer:")
    print("    [1] DeiT-tiny  (Vision Transformer)  ->  clasifica senal")
    print("    [2] Flan-T5    (Text  Transformer)  ->  explica senal")
    print("=" * 62 + "\n")

    # ── Cargar los dos Transformers ───────────────────────────────────────────
    vit = ViTClassifier()   # Transformer 1: vision
    t5  = T5Explainer()     # Transformer 2: lenguaje

    # ── Detector de color (preprocesamiento, no es un transformer) ───────────
    mode     = "both" if (args.video or not args.image) else "color"
    detector = TrafficSignDetector(conf_threshold=0.35, mode=mode,
                                   min_area=MIN_AREA_AGENT)

    # ── Crear el agente ───────────────────────────────────────────────────────
    agent = TrafficAgent(detector, vit, t5)

    print("Agente listo.\n")

    # ── Ejecutar segun el modo ────────────────────────────────────────────────
    if args.image:
        run_image(args.image, agent)
    elif args.video:
        run_video(args.video, agent)
    else:
        run_video(0, agent)     # Webcam por defecto


if __name__ == "__main__":
    main()
