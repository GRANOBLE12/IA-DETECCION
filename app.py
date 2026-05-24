"""
app.py — Deteccion y clasificacion de senales de transito en tiempo real.

Pipeline:
    Frame de video
        -> YOLOv8 detecta DONDE estan las senales (bounding boxes)
        -> DeiT-tiny (entrenado localmente) clasifica QUE senal es (43 clases GTSRB)
        -> Muestra nombre + descripcion en pantalla

Uso:
    python app.py                         # Webcam
    python app.py --image foto.jpg        # Foto con varias senales
    python app.py --video ruta/video.mp4  # Archivo de video
    python app.py --no-detector           # Sin YOLOv8, apunta la camara a la senal
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTForImageClassification

from labels import CLASS_NAMES, DESCRIPTIONS

# ─── Configuracion ───────────────────────────────────────────────────────────

MODEL_ARCH  = "facebook/deit-tiny-patch16-224"   # arquitectura base
MODEL_PATH  = "vit_gtsrb.pth"                    # pesos entrenados localmente
NUM_CLASSES = 43
IMAGE_SIZE  = 224
CONF_CLS    = 0.3    # confianza minima para mostrar resultado

# Colores BGR
COLOR_BOX   = (0, 200, 255)
COLOR_LABEL = (255, 255, 255)
COLOR_BG    = (0, 130, 200)
COLOR_WARN  = (0, 60, 255)


# ─── Clasificador ViT ────────────────────────────────────────────────────────

class ViTClassifier:
    """
    Clasificador usando el modelo DeiT-tiny entrenado localmente en GTSRB.
    Carga los pesos de vit_gtsrb.pth (generado por train_vit.py).
    """

    TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: No se encontro '{MODEL_PATH}'.")
            print("Primero ejecuta:  python train_vit.py")
            sys.exit(1)

        print(f"Cargando modelo DeiT-tiny desde '{MODEL_PATH}'...")
        self.model = ViTForImageClassification.from_pretrained(
            MODEL_ARCH,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        self.model.eval()
        print("Modelo listo. (99.05% precision en GTSRB)")

    @torch.no_grad()
    def classify(self, crop_bgr: np.ndarray) -> tuple[int, str, float]:
        """
        Clasifica un recorte de imagen BGR.

        Returns:
            (class_id, class_name, confidence)
        """
        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.TRANSFORM(rgb).unsqueeze(0)   # (1, 3, 224, 224)

        logits   = self.model(pixel_values=tensor).logits
        probs    = F.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

        class_id   = idx.item()
        confidence = conf.item()
        class_name = CLASS_NAMES.get(class_id, f"Clase {class_id}")
        return class_id, class_name, confidence


# ─── Dibujo en pantalla ──────────────────────────────────────────────────────

def draw_detection(frame, box, class_name, conf_cls):
    x1, y1, x2, y2 = box
    color = COLOR_BOX if conf_cls >= CONF_CLS else COLOR_WARN

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{class_name}  {conf_cls:.0%}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    ty = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
    cv2.putText(frame, label, (x1 + 2, ty - 2), font, scale, COLOR_LABEL, thickness, cv2.LINE_AA)
    return frame


def draw_panel(frame, detections_info):
    """Panel lateral con descripciones de las senales detectadas."""
    if not detections_info:
        return frame

    h, w    = frame.shape[:2]
    panel_w = 360
    panel   = np.full((h, panel_w, 3), 30, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y    = 30
    cv2.putText(panel, "Senales detectadas:", (10, y), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    y += 25

    for class_id, class_name, conf in detections_info[:4]:
        if conf < CONF_CLS:
            continue

        cv2.putText(panel, f">> {class_name}", (10, y), font, 0.5, COLOR_BOX, 1, cv2.LINE_AA)
        y += 20

        desc  = DESCRIPTIONS.get(class_id, "")
        words = desc.split()
        line  = ""
        for word in words:
            test = line + word + " "
            (tw, _), _ = cv2.getTextSize(test, font, 0.4, 1)
            if tw > panel_w - 20:
                cv2.putText(panel, line.strip(), (15, y), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
                y    += 16
                line  = word + " "
            else:
                line = test
        if line.strip():
            cv2.putText(panel, line.strip(), (15, y), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        y += 25

        cv2.line(panel, (10, y), (panel_w - 10, y), (60, 60, 60), 1)
        y += 12
        if y > h - 30:
            break

    return np.hstack([frame, panel])


# ─── Bucle principal ─────────────────────────────────────────────────────────

def run_image(args, classifier, detector):
    """Procesa una imagen estatica con muchas senales."""
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: no se pudo abrir '{args.image}'")
        sys.exit(1)

    print(f"Procesando imagen: {args.image}")
    detections_info = []

    if detector is not None:
        dets = detector.detect(frame)
        print(f"Regiones detectadas: {len(dets)}")
        if not dets:
            print("No se detectaron senales. Prueba con --no-detector para clasificar la imagen completa.")
        for det in dets:
            crop = detector.crop(frame, det["box"])
            class_id, class_name, conf = classifier.classify(crop)
            detections_info.append((class_id, class_name, conf))
            draw_detection(frame, det["box"], class_name, conf)
            print(f"  Detectada: {class_name}  ({conf:.0%})")
    else:
        # Sin detector: clasifica la imagen completa
        class_id, class_name, conf = classifier.classify(frame)
        detections_info.append((class_id, class_name, conf))
        cv2.putText(frame, f"{class_name}  {conf:.0%}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BOX, 2, cv2.LINE_AA)

    display = draw_panel(frame, detections_info)

    # Redimensionar si la imagen es muy grande
    h, w = display.shape[:2]
    max_h, max_w = 900, 1400
    if h > max_h or w > max_w:
        scale   = min(max_h / h, max_w / w)
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    cv2.namedWindow("GTSRB - Imagen", cv2.WINDOW_NORMAL)
    cv2.imshow("GTSRB - Imagen", display)
    print("Presiona cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run(args):
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Dispositivo: {device_name}")

    # Cargar clasificador ViT
    classifier = ViTClassifier()

    # Cargar detector (por defecto: detector de color rojo)
    detector = None
    if not args.no_detector:
        from detector import TrafficSignDetector
        mode     = "both" if args.video or not args.image else "color"
        detector = TrafficSignDetector(conf_threshold=0.35, mode=mode)

    # ── Modo imagen ───────────────────────────────────────────────────────────
    if args.image:
        run_image(args, classifier, detector)
        return

    # Abrir fuente de video
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: no se pudo abrir '{args.video}'")
            sys.exit(1)
        print(f"Reproduciendo: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: no se pudo abrir la webcam.")
            sys.exit(1)
        print("Webcam activa. Presiona 'q' para salir.")

    cv2.namedWindow("GTSRB - Deteccion de Senales", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections_info = []

        if detector is not None:
            # Modo detector (YOLOv8) + clasificador (ViT)
            dets = detector.detect(frame)
            for det in dets:
                crop = detector.crop(frame, det["box"])
                class_id, class_name, conf = classifier.classify(crop)
                detections_info.append((class_id, class_name, conf))
                draw_detection(frame, det["box"], class_name, conf)

        else:
            # Modo directo: recuadro en el centro para apuntar la senal
            h, w   = frame.shape[:2]
            cx, cy = w // 2, h // 2
            size   = min(w, h) // 2
            x1c    = cx - size // 2
            y1c    = cy - size // 2
            x2c    = cx + size // 2
            y2c    = cy + size // 2

            crop = frame[y1c:y2c, x1c:x2c]
            cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), COLOR_BOX, 2)
            cv2.putText(frame, "Pon la senal aqui", (x1c, y1c - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BOX, 1, cv2.LINE_AA)

            class_id, class_name, conf = classifier.classify(crop)
            detections_info.append((class_id, class_name, conf))

            if conf >= CONF_CLS:
                cv2.putText(frame, f"{class_name}  {conf:.0%}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BOX, 2, cv2.LINE_AA)

        display = draw_panel(frame, detections_info)
        cv2.imshow("GTSRB - Deteccion de Senales", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── Argumentos ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Deteccion de senales de transito")
    p.add_argument("--image",       type=str,            help="Ruta a una foto con senales (jpg, png...)")
    p.add_argument("--video",       type=str,            help="Ruta al video (sin esto = webcam)")
    p.add_argument("--no-detector", action="store_true", help="Sin YOLOv8, clasifica el centro del frame")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
