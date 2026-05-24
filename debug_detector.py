"""
debug_detector.py — Diagnostica el detector de color en una imagen.

Uso:
    python debug_detector.py ruta/imagen.jpg

Muestra:
  - La mascara de color (lo que detecta el filtro HSV)
  - Los bounding boxes detectados en la imagen original
  - Estadisticas de los valores HSV encontrados

Util para ajustar parametros si el detector no encuentra senales.
"""

import sys
import cv2
import numpy as np


def debug(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: no se pudo abrir '{image_path}'")
        sys.exit(1)

    h_img, w_img = frame.shape[:2]
    print(f"Imagen: {w_img}x{h_img} px")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mismos rangos que detector.py actualizado
    mask_r1 = cv2.inRange(hsv, np.array([0,   40, 40]),  np.array([12,  255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([158, 40, 40]),  np.array([180, 255, 255]))
    mask_y  = cv2.inRange(hsv, np.array([15,  60, 70]),  np.array([42,  255, 255]))
    mask_b  = cv2.inRange(hsv, np.array([95,  80, 60]),  np.array([135, 255, 255]))

    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    mask   = cv2.bitwise_or(mask_r, mask_y)
    mask   = cv2.bitwise_or(mask, mask_b)

    pct_r = mask_r.sum() / 255 / (h_img * w_img) * 100
    pct_y = mask_y.sum() / 255 / (h_img * w_img) * 100
    pct_b = mask_b.sum() / 255 / (h_img * w_img) * 100
    print(f"Pixeles rojos  : {pct_r:.1f}%")
    print(f"Pixeles amarill: {pct_y:.1f}%")
    print(f"Pixeles azules : {pct_b:.1f}%")

    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_p = cv2.dilate(mask, kernel, iterations=2)
    mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_CLOSE, kernel)

    # Contornos
    contours, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nContornos encontrados: {len(contours)}")

    MAX_AREA = h_img * w_img * 0.6
    MIN_AREA = 120
    validos = 0
    frame_out = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / max(h, 1)
        pasa_area  = MIN_AREA <= area <= MAX_AREA
        pasa_ratio = 0.18 <= ratio <= 5.0
        color = (0, 255, 0) if (pasa_area and pasa_ratio) else (0, 0, 255)
        cv2.rectangle(frame_out, (x, y), (x+w, y+h), color, 1)
        msg = f"area={int(area)} ratio={ratio:.2f}"
        if pasa_area and pasa_ratio:
            validos += 1
            cv2.putText(frame_out, msg, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        else:
            razon = []
            if not pasa_area:  razon.append(f"area<{MIN_AREA}" if area < MIN_AREA else f"area>{MAX_AREA:.0f}")
            if not pasa_ratio: razon.append(f"ratio={ratio:.2f}")
            print(f"  Rechazado: {msg} -> {', '.join(razon)}")

    print(f"Detecciones validas: {validos}")

    # Mostrar resultados
    mask_vis = cv2.cvtColor(mask_p, cv2.COLOR_GRAY2BGR)
    lado = min(800, w_img)
    scale = lado / w_img
    dim = (int(w_img * scale), int(h_img * scale))
    vis = np.hstack([
        cv2.resize(frame_out,  dim),
        cv2.resize(mask_vis,   dim),
    ])

    cv2.imshow("Debug detector | Izq: detecciones (verde=OK, rojo=rechazado) | Der: mascara color", vis)
    print("\nPresiona cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python debug_detector.py ruta/imagen.jpg")
        sys.exit(1)
    debug(sys.argv[1])
