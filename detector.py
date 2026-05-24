"""
detector.py — Detecta senales de transito en imagenes/frames.

Dos detectores disponibles:
  1. ColorDetector  — detecta senales rojas por color (funciona siempre, sin modelos)
  2. YoloDetector   — YOLOv8 COCO (solo detecta STOP, usado como complemento)

El ColorDetector es el principal porque las senales colombianas/alemanas
son predominantemente rojas y el ViT luego clasifica cada region.
"""

import cv2
import numpy as np


# ─── Detector por color (principal) ──────────────────────────────────────────

class ColorDetector:
    """
    Detecta senales de transito buscando regiones rojas en la imagen.

    Funciona sin ningun modelo externo. Las senales regulatorias y preventivas
    son predominantemente rojas, lo que permite encontrarlas por color HSV.
    Luego el ViT clasifica que tipo de senal es cada region.
    """

    def __init__(self, min_area: int = 120, max_area_ratio: float = 0.6):
        """
        Args:
            min_area:       Area minima en pixeles para considerar una senal.
                            120 permite detectar senales pequenas en collages/fotos
                            con senales de ~40x40 pixeles (borde rojo ~120 px²).
            max_area_ratio: Fraccion maxima del frame que puede ocupar una senal.
        """
        self.min_area       = min_area
        self.max_area_ratio = max_area_ratio

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detecta senales de transito por color (rojo, amarillo, azul, blanco).

        Args:
            frame: Imagen BGR de OpenCV.

        Returns:
            Lista de detecciones [{box, conf, label}, ...]
        """
        h_img, w_img = frame.shape[:2]
        max_area      = h_img * w_img * self.max_area_ratio

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rojo (dos rangos en HSV) — saturacion baja a 40 para rojos menos saturados
        mask_r1 = cv2.inRange(hsv, np.array([0,   40, 40]),  np.array([12,  255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([158, 40, 40]),  np.array([180, 255, 255]))
        # Amarillo/naranja (senales preventivas y de obras) — incluye amarillo palido
        mask_y  = cv2.inRange(hsv, np.array([15,  60, 70]),  np.array([42,  255, 255]))
        # Azul (senales informativas)
        mask_b  = cv2.inRange(hsv, np.array([95,  80, 60]),  np.array([135, 255, 255]))

        mask = cv2.bitwise_or(mask_r1, mask_r2)
        mask = cv2.bitwise_or(mask, mask_y)
        mask = cv2.bitwise_or(mask, mask_b)

        # Kernel pequeno: no destruye bordes finos de triangulos/circulos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask   = cv2.dilate(mask, kernel, iterations=2)   # une pixeles cercanos
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # cierra huecos

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Filtrar formas muy alargadas (triangulos pueden tener ratio 0.5-2.0)
            ratio = w / h
            if ratio > 5.0 or ratio < 0.18:
                continue

            # Margen para capturar el interior completo de la senal
            margin = int(max(w, h) * 0.12)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w_img, x + w + margin)
            y2 = min(h_img, y + h + margin)

            detections.append({
                "box":   [x1, y1, x2, y2],
                "conf":  0.85,
                "label": "traffic_sign",
            })

        # Eliminar detecciones que se solapan mucho (NMS simple)
        detections = self._nms(detections, iou_thresh=0.3)
        return detections

    def _nms(self, detections: list[dict], iou_thresh: float) -> list[dict]:
        """Non-Maximum Suppression: elimina bounding boxes solapados."""
        if not detections:
            return []

        boxes = np.array([d["box"] for d in detections], dtype=float)
        scores = np.array([d["conf"] for d in detections])
        order  = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i  = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j  = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                      (boxes[order[1:], 3] - boxes[order[1:], 1])
            iou     = inter / (area_i + area_j - inter + 1e-6)

            order = order[1:][iou < iou_thresh]

        return [detections[i] for i in keep]

    def crop(self, frame: np.ndarray, box: list[int], margin: int = 0) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, box[0] - margin)
        y1 = max(0, box[1] - margin)
        x2 = min(w, box[2] + margin)
        y2 = min(h, box[3] + margin)
        return frame[y1:y2, x1:x2].copy()


# ─── Detector YOLO (complemento para escenas reales) ─────────────────────────

class YoloDetector:
    """
    Detector YOLOv8 COCO. Detecta stop signs en escenas reales de video.
    Se usa como complemento del ColorDetector en modo video/webcam.
    """

    def __init__(self, conf_threshold: float = 0.35):
        from ultralytics import YOLO
        print("Cargando YOLOv8...")
        self.model = YOLO("yolov8n.pt")
        self.conf  = conf_threshold
        print("YOLOv8 listo.")

    def detect(self, frame: np.ndarray) -> list[dict]:
        results    = self.model.predict(source=frame, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                if w < 20 or h < 20:
                    continue
                detections.append({
                    "box":   [x1, y1, x2, y2],
                    "conf":  float(box.conf[0]),
                    "label": self.model.names[int(box.cls[0])],
                })
        return detections

    def crop(self, frame: np.ndarray, box: list[int], margin: int = 8) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, box[0] - margin)
        y1 = max(0, box[1] - margin)
        x2 = min(w, box[2] + margin)
        y2 = min(h, box[3] + margin)
        return frame[y1:y2, x1:x2].copy()


# ─── Selector automatico ─────────────────────────────────────────────────────

class TrafficSignDetector:
    """
    Selector inteligente de detector:
      - Imagenes/fotos  -> ColorDetector  (encuentra todas las senales rojas)
      - Video/webcam    -> ColorDetector + YoloDetector combinados
    """

    def __init__(self, conf_threshold: float = 0.35, mode: str = "color",
                 min_area: int = 120):
        """
        Args:
            mode:     "color" (solo color), "yolo" (solo YOLO), "both" (ambos)
            min_area: Area minima de region para considerarla senal candidata.
                      Subir este valor reduce falsos positivos en fondos de color.
        """
        self.color_det = ColorDetector(min_area=min_area)
        self.yolo_det  = None
        self.mode      = mode

        if mode in ("yolo", "both"):
            try:
                self.yolo_det = YoloDetector(conf_threshold)
            except Exception as e:
                print(f"YOLO no disponible: {e}. Usando solo detector de color.")
                self.mode = "color"

    def detect(self, frame: np.ndarray) -> list[dict]:
        dets = self.color_det.detect(frame)

        if self.mode == "both" and self.yolo_det:
            yolo_dets = self.yolo_det.detect(frame)
            # Agregar detecciones YOLO que no se solapan con las de color
            for yd in yolo_dets:
                overlap = any(self._iou(yd["box"], d["box"]) > 0.3 for d in dets)
                if not overlap:
                    dets.append(yd)

        return dets

    def crop(self, frame, box, margin=0):
        return self.color_det.crop(frame, box, margin)

    @staticmethod
    def _iou(b1, b2):
        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)
