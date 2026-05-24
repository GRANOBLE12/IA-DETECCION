# Informe Técnico: Sistema de Reconocimiento y Clasificación de Señales de Tránsito
## Inteligencia Artificial — Tercer Corte

**Estudiante:** Andres Felipe Bolaños Zuñiga  
**Fecha:** Mayo 2026  
**Repositorio:** `gtsrb/`  

---

## 1. Descripción General del Proyecto

Este proyecto implementa un **sistema completo de reconocimiento de señales de tránsito en tiempo real** usando dos arquitecturas transformer conectadas por un agente reactivo autónomo.

El sistema se compone de tres módulos ejecutables:

| Módulo | Archivo | Función |
|---|---|---|
| **Agente reactivo** | `agente.py` | **Módulo principal** — conecta ViT + Flan-T5 automáticamente |
| App visual | `app.py` | Detecta y etiqueta señales en webcam/video/imagen |
| Asistente Q&A | `assistant.py` | Responde preguntas de texto sobre señales |

**El requisito central del proyecto** — que un transformer genere información contextual a partir de la salida de una red neuronal — se cumple en `agente.py`:

```
imagen → DeiT-tiny (Transformer 1) → "STOP, 99.7%" → Flan-T5 (Transformer 2) → "Detencion obligatoria..."
```

---

## 2. Dataset Utilizado: GTSRB

### ¿Qué es GTSRB?

El **German Traffic Sign Recognition Benchmark (GTSRB)** es uno de los datasets de señales de tránsito más reconocidos en visión por computador. Fue creado por el Institut für Neuroinformatik (INI) de la Universidad de Bochum, Alemania, y fue utilizado en una competencia internacional en 2011.

### Características del dataset

| Característica | Valor |
|---|---|
| Número de clases | **43 categorías** de señales |
| Imágenes de entrenamiento | ~**39,209 imágenes** |
| Imágenes de prueba | ~**12,630 imágenes** |
| Tamaño de imágenes | Variable (30×30 hasta 250×250 px) |
| Formato | PPM (Portable Pixmap), RGB |
| Descarga automática | Sí, via `torchvision.datasets.GTSRB` (~340 MB) |

### Las 43 clases

- **Clases 0–8**: Límites de velocidad (20, 30, 50, 60, 70, 80, 100, 120 km/h)
- **Clases 9–17**: Señales de prohibición (no adelantar, prohibido el paso, STOP, etc.)
- **Clases 18–31**: Señales de peligro y advertencia (curvas, pavimento irregular, peatones, animales, etc.)
- **Clases 32–42**: Señales de obligación y fin de restricciones (girar, seguir recto, rotonda, etc.)

### ¿Por qué GTSRB y no un dataset colombiano?

No existe un dataset público de señales colombianas de tamaño suficiente para entrenar redes neuronales. Colombia sigue la **Convención de Viena sobre señalización vial**, lo que hace que las señales alemanas sean equivalentes funcionales a las colombianas:

- La señal de STOP es idéntica internacionalmente (octágono rojo, texto blanco)
- Los límites de velocidad usan el mismo formato circular con borde rojo
- Las señales de advertencia usan el triángulo rojo
- Las señales de obligación usan el círculo azul

**Validación en tiempo real:** el modelo identificó correctamente la señal colombiana "PARE" (mostrada en pantalla de teléfono) con **99.7% de confianza**, confirmando la compatibilidad.

---

## 3. Arquitectura del Sistema

### 3.1 Diagrama del Agente (módulo principal)

```
╔══════════════════════════════════════════════════════════════════╗
║              AGENTE REACTIVO  (agente.py)                       ║
║                                                                  ║
║  PERCIBIR       CLASIFICAR          RAZONAR          ACTUAR      ║
║                                                                  ║
║  Frame BGR  ->  ColorDetector  ->  Senal nueva?  ->  Flan-T5   ║
║  (webcam /      (HSV masks)      Confianza >=97%   (Transformer ║
║   video /           |            6 frames           2 - texto)  ║
║   imagen)      DeiT-tiny         consecutivos?           |      ║
║                (Transformer 1         |            Explicacion  ║
║                 - vision)        SI -> actuar      automatica   ║
║                     |            NO -> esperar     en pantalla  ║
║               class_id (0-42)                                   ║
║               + confianza %                                      ║
╚══════════════════════════════════════════════════════════════════╝
```

### 3.2 Diagrama completo del proyecto

```
                        ENTRADA
          Webcam / Video / Imagen / Pregunta de texto
               |                |              |
         agente.py           app.py      assistant.py
         AGENTE           Deteccion       Asistente
         REACTIVO          visual           Q&A
               |                |              |
        detector.py       detector.py    Palabras clave
        ColorDetector     ColorDetector   (KEYWORDS dict)
               |                |              |
        [Transformer 1]  [Transformer 1] [Transformer 2]
         DeiT-tiny ViT    DeiT-tiny ViT   Flan-T5-base
         -> class_id      -> class_name   -> respuesta
               |
        [Transformer 2]
         Flan-T5-base
         -> explicacion
           automatica
               |
          labels.py
   (43 clases: nombres + descripciones)
```

---

## 4. Componente 1: Entrenamiento del Clasificador Visual

### Archivo: `train_vit.py`

#### ¿Qué modelo se usa y por qué?

Se eligió **DeiT-tiny** (`facebook/deit-tiny-patch16-224`), una variante pequeña del **Data-efficient Image Transformer**.

**¿Por qué un Vision Transformer y no una CNN clásica?**

Los Transformers aplican el mecanismo de **self-attention** — originalmente diseñado para procesamiento de lenguaje — a imágenes. En lugar de procesar píxeles localmente con filtros convolucionales, un ViT:

1. Divide la imagen en **parches (patches)** de 16×16 píxeles
2. Convierte cada parche en un vector (embedding)
3. Aplica **atención multi-cabeza** entre todos los parches simultáneamente
4. Aprende qué partes de la imagen son relevantes entre sí

Para una imagen de 224×224 px → 196 parches de 16×16, cada uno procesado como un "token" (análogo a una palabra en NLP).

**¿Por qué DeiT-tiny específicamente?**

| Modelo | Parámetros | Memoria | Tiempo/batch CPU | Precisión GTSRB |
|---|---|---|---|---|
| ViT-base | 86M | ~350 MB | ~40 seg | >99% |
| DeiT-small | 22M | ~90 MB | ~10 seg | ~98% |
| **DeiT-tiny** | **5.7M** | **~25 MB** | **~2 seg** | **~99%** |

DeiT-tiny fue diseñado para ser **eficiente en entrenamiento sin GPU**, usando destilación de conocimiento desde modelos más grandes. Con 2 segundos por batch, 2 épocas completas toman ~50 minutos en CPU.

#### Técnica: Transfer Learning + Fine-tuning

El modelo **no se entrena desde cero**. Se parte de pesos pre-entrenados en **ImageNet-1k** (1.28 millones de imágenes, 1000 clases) y se adaptan a GTSRB (43 clases).

**¿Por qué transfer learning?**

- ImageNet enseña al modelo a reconocer bordes, texturas, formas y objetos genéricos
- Ese conocimiento es directamente útil para señales de tránsito (formas circulares, triangulares, colores)
- Entrenar desde cero requeriría órdenes de magnitud más datos y tiempo

**Ajuste de la cabeza de clasificación:**

```python
model = ViTForImageClassification.from_pretrained(
    "facebook/deit-tiny-patch16-224",
    num_labels=43,                    # 43 clases GTSRB (original: 1000 ImageNet)
    ignore_mismatched_sizes=True,     # Reemplaza la capa final
)
```

Solo la capa de clasificación final se reinicializa; el resto del transformer mantiene los pesos de ImageNet.

#### Hiperparámetros de entrenamiento

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Épocas | 2 | Suficiente con transfer learning; más épocas generan sobreajuste |
| Batch size | 32 | Balance entre velocidad y estabilidad del gradiente |
| Learning rate | 2×10⁻⁴ | Bajo para fine-tuning (no destruir pesos pre-entrenados) |
| Optimizador | AdamW | Adam con regularización L2 (weight decay=0.01) |
| Scheduler | CosineAnnealingLR | Reduce LR gradualmente siguiendo una curva coseno |
| Hilos CPU | 4 | Limita uso de CPU para no saturar el sistema |

#### Data Augmentation

```python
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # variacion de luz
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],            # media ImageNet
                         [0.229, 0.224, 0.225]),            # desv. estandar ImageNet
])
```

El `ColorJitter` simula variaciones de iluminación real, haciendo el modelo más robusto.

#### Resultados del entrenamiento

```
Epoca 1:  Train: 91.2%  |  Val: 98.6%  |  Tiempo: ~25 min
Epoca 2:  Train: 99.6%  |  Val: 99.8%  |  Tiempo: ~25 min
───────────────────────────────────────────────────────
TEST FINAL:  99.05%  (sobre 12,630 imagenes no vistas)
```

**¿Por qué 99% en solo 2 épocas?**

GTSRB tiene señales con poca variabilidad intra-clase (objetos manufacturados, siempre iguales) y alta variabilidad inter-clase (cada clase luce muy diferente). La diferencia entre train (99.6%) y test (99.05%) es de solo 0.55%, lo que confirma que **no hay sobreajuste** — el modelo genuinamente aprendió las señales.

---

## 5. Componente 2: Detección de Señales

### Archivo: `detector.py`

La detección responde a **¿DÓNDE están las señales?** antes de que el ViT clasifique **¿QUÉ señal es?**.

#### Enfoque primario: Detector por Color (ColorDetector)

**Principio:** Las señales de tránsito tienen colores regulados por ley — rojo, amarillo, azul. Se detectan filtrando esos colores en el espacio **HSV** (Hue-Saturation-Value).

**¿Por qué HSV y no RGB?**

En RGB el rojo varía con la iluminación: (200,10,10) con luz natural, (120,60,40) con poca luz. En HSV el "rojo" siempre está en H=[0°,12°] o H=[348°,360°] independientemente de la iluminación.

```python
# Rojo — dos rangos porque "wrappea" el circulo de color HSV
mask_r1 = cv2.inRange(hsv, [0,  40, 40], [12,  255, 255])   # rojo en 0 grados
mask_r2 = cv2.inRange(hsv, [158,40, 40], [180, 255, 255])   # rojo en 360 grados

# Amarillo (senales preventivas)
mask_y  = cv2.inRange(hsv, [15, 60, 70], [42,  255, 255])

# Azul (senales informativas y de obligacion)
mask_b  = cv2.inRange(hsv, [95, 80, 60], [135, 255, 255])
```

**Pipeline de detección:**

1. Filtro HSV → máscara binaria (blanco = color de señal)
2. Dilatación morfológica (kernel 3×3, 2 iteraciones) → une píxeles dispersos
3. Cierre morfológico → rellena huecos internos
4. Búsqueda de contornos → regiones conectadas
5. Filtros geométricos: área mínima, área máxima (60% del frame), ratio ancho/alto (0.18–5.0)
6. Non-Maximum Suppression (NMS, IoU > 0.3) → elimina bounding boxes solapados

**¿Por qué kernel 3×3 y no 7×7?**

Los bordes de señales triangulares son delgados (~5-10 px). Un kernel de 7×7 los erosiona y destruye antes de encontrar los contornos. Con 3×3 se mantiene la integridad del borde.

**Parámetro `min_area` configurable por módulo:**

```python
class TrafficSignDetector:
    def __init__(self, conf_threshold=0.35, mode="color", min_area=120):
        self.color_det = ColorDetector(min_area=min_area)
```

El agente usa `min_area=600` (más estricto) para evitar detectar manchas de luz o reflejos en paredes. La app usa `min_area=120` para mayor sensibilidad en fotos.

#### Enfoque complementario: YOLOv8 (YoloDetector)

**YOLOv8** (You Only Look Once v8) es un detector de objetos en tiempo real basado en redes convolucionales profundas. Se usa el modelo `yolov8n.pt` (nano), el más ligero.

**Limitación:** YOLOv8n se entrenó en COCO (80 clases generales) y solo reconoce señales de STOP. Por eso se usa únicamente como complemento del detector de color en modo video.

---

## 6. Componente 3: Agente Reactivo Dual-Transformer

### Archivo: `agente.py` — Módulo principal del proyecto

Este es el componente central que cumple el requisito académico: **conecta automáticamente el Vision Transformer con el Text Transformer** en un ciclo agente percibir → razonar → actuar.

#### Parámetros del agente

| Parámetro | Valor | Justificación |
|---|---|---|
| `CONF_MIN` | **0.97 (97%)** | Solo actúa cuando el ViT está casi completamente seguro — elimina falsos positivos de paredes y fondos |
| `MIN_AREA_AGENT` | **600 px²** | Requiere regiones de al menos 600 px² — descarta manchas de luz y ruido de cámara |
| `TRIGGER_FRAMES` | **6 frames** | La señal debe aparecer en 6 frames consecutivos antes de activar Flan-T5 — evita activaciones por señales en movimiento |

#### Transformer 1: ViTClassifier (DeiT-tiny)

```python
class ViTClassifier:
    @torch.no_grad()
    def classify(self, crop_bgr):
        rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.TRANSFORM(rgb).unsqueeze(0)        # (1, 3, 224, 224)
        logits = self.model(pixel_values=tensor).logits  # (1, 43)
        probs  = F.softmax(logits, dim=1)                # probabilidades
        conf, idx = probs.max(dim=1)
        return idx.item(), CLASS_NAMES[idx.item()], conf.item()
```

#### Transformer 2: T5Explainer (Flan-T5-base)

Recibe el `class_id` del ViT y genera automáticamente una explicación en español:

```python
class T5Explainer:
    def explain(self, class_id):
        nombre = CLASS_NAMES[class_id]
        desc   = DESCRIPTIONS[class_id]

        # Prompt formato Q&A — evita que T5 repita la instruccion
        prompt = (
            f"Question: What does the traffic sign '{nombre}' mean?\n"
            f"Context: {desc}\n"
            f"Answer in Spanish:"
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,               # Beam search: 4 hipotesis en paralelo
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**¿Por qué formato Q&A?** El formato instrucción directa ("You are a driving instructor...") hacía que T5 repitiera partes del prompt en la respuesta. El formato Q&A (pregunta + contexto + "Answer in Spanish:") produce respuestas directas sin artefactos.

#### Lógica del agente (percibir → razonar → actuar)

```python
def perceive(self, frame):
    # Detectar regiones de color + clasificar con ViT
    dets = self.detector.detect(frame)
    classified = [(vit.classify(crop), box) for crop in dets]

    # FILTRO CLAVE: solo la mejor deteccion (mayor confianza)
    # Evita multiples cajas superpuestas de la misma senal
    best = max(classified, key=lambda x: x[0][2])
    return [best] if best[0][2] >= CONF_MIN else []

def reason_and_act(self, detections):
    if not detections: return
    cid, name, conf = detections[0]

    # Razonar: es la misma senal de antes?
    if cid == self.last_class_id:
        self.consecutive += 1
    else:
        self.consecutive = 1
        self.last_class_id = cid

    # Actuar: solo cuando hay suficientes frames consecutivos
    if self.consecutive == TRIGGER_FRAMES and not self.generating:
        threading.Thread(target=self.t5.explain, args=(cid,)).start()
```

#### Normalización de texto para OpenCV

OpenCV no soporta caracteres UTF-8 (tildes, ñ) — los muestra como `??`. Se aplica una función de normalización antes de dibujar cualquier texto:

```python
def _limpiar(texto):
    tabla = str.maketrans("áéíóúÁÉÍÓÚñÑ¿¡", "aeiouAEIOUnN  ")
    return texto.translate(tabla)
```

#### Threading no bloqueante

Flan-T5 tarda ~3-9 segundos en CPU. Si se ejecutara en el hilo principal, el video se congelaría. Se usa un **hilo de fondo**:

```python
def _act(self, class_id, sign_name):
    self.generating = True

    def _run_t5():
        text = self.t5.explain(class_id)   # ~3-9 seg en CPU
        with self._lock:
            self.explanation = text         # Actualiza el panel
        self.generating = False

    threading.Thread(target=_run_t5, daemon=True).start()
    # El hilo principal sigue capturando frames normalmente
```

---

## 7. Componente 4: Aplicación Visual

### Archivo: `app.py`

Versión simplificada sin generación de texto. Para cada frame:

1. ColorDetector → bounding boxes
2. DeiT-tiny → (class_id, class_name, confidence) por cada box
3. Dibuja caja + etiqueta + panel lateral con descripción de `labels.py`

**Umbral de confianza:** CONF_CLS = 0.30 (más bajo que el agente — muestra más detecciones)

---

## 8. Componente 5: Asistente Conversacional

### Archivo: `assistant.py`

Arquitectura híbrida **Recuperación + Generación** (similar a RAG):

1. **Recuperación por palabras clave** (KEYWORDS dict) → encuentra clase relevante
2. **Prompt estructurado** con nombre y descripción de la señal
3. **Flan-T5-base genera** respuesta en español
4. **Validación + fallback** a `labels.py` si T5 genera texto corrupto

---

## 9. Base de Conocimiento

### Archivo: `labels.py`

43 clases con nombre y descripción en español, usadas por todos los módulos:

```python
CLASS_NAMES  = {14: "STOP", 13: "Ceda el paso", 27: "Peatones", ...}
DESCRIPTIONS = {14: "Detencion obligatoria. Para completamente antes de continuar.", ...}
```

---

## 10. Tecnologías Utilizadas

| Librería | Versión | Uso | Justificación |
|---|---|---|---|
| **PyTorch** | 2.12.0+cpu | Motor de deep learning | Framework estándar para IA |
| **torchvision** | ≥0.15 | Datasets, transforms | Carga automática de GTSRB |
| **transformers** | 5.9.0 | DeiT-tiny + Flan-T5 | HuggingFace: modelos pre-entrenados |
| **OpenCV** | ≥4.8 | Video, dibujo | Biblioteca estándar de visión |
| **NumPy** | ≥1.24 | Operaciones matriciales | Base numérica del stack |
| **ultralytics** | ≥8.0 | YOLOv8 | Detector en tiempo real |
| **Pillow** | ≥9.0 | Conversión de imágenes | Requerido por torchvision |
| **tqdm** | ≥4.65 | Barras de progreso | Seguimiento del entrenamiento |

---

## 11. Flujo Completo del Agente

```
Inicio: python agente.py
    |
    v
Cargar Transformer 1: DeiT-tiny (vit_gtsrb.pth, 5.7M params)
Cargar Transformer 2: Flan-T5-base (google/flan-t5-base, 250M params)
    |
    v
[Bucle por cada frame]
    |
    +-- PERCIBIR: cv2.VideoCapture → frame BGR
    |        |
    |   ColorDetector.detect(frame)
    |        | → mascaras HSV (rojo/amarillo/azul)
    |        | → morfologia (kernel 3x3)
    |        | → contornos + NMS
    |        | → bounding boxes (min_area >= 600 px2)
    |        |
    |   Para cada box:
    |     crop = frame[y1:y2, x1:x2]
    |     BGR → RGB → Resize 224x224 → Normalize
    |     DeiT-tiny forward pass
    |     softmax → (class_id, confianza)
    |        |
    |   Filtrar: quedarse solo con la de MAYOR confianza
    |   Si confianza < 97% → descartar
    |
    +-- RAZONAR:
    |   ¿Es la misma clase que el frame anterior?
    |     SI → consecutive += 1
    |     NO → consecutive = 1, guardar nueva clase
    |   ¿consecutive == 6 Y no hay T5 corriendo?
    |     NO → mostrar frame, continuar
    |     SI → ACTUAR
    |
    +-- ACTUAR: lanzar Flan-T5 en hilo de fondo
    |   Prompt: "Question: What does '{nombre}' mean? Context: {desc} Answer in Spanish:"
    |   model.generate(num_beams=4, max_new_tokens=80)
    |   _limpiar(texto) → reemplaza tildes para OpenCV
    |   Escribe en self.explanation (visible en panel)
    |
    +-- VISUALIZAR:
        Una sola caja naranja sobre la señal
        Panel lateral: nombre, confianza, barra, explicacion de T5
        cv2.imshow()
```

---

## 12. Retos y Soluciones

### Reto 1: ViT-base demasiado lento en CPU

**Problema:** ViT-base (86M parámetros) tardaba ~40 segundos por batch → más de 40 horas para 2 épocas.  
**Solución:** Cambiar a **DeiT-tiny** (5.7M parámetros) → ~2 segundos por batch → 50 minutos para 2 épocas.

### Reto 2: Detector no encontraba señales triangulares

**Problema:** El detector de color retornaba 0 detecciones en señales triangulares de advertencia.  
**Causa:** El kernel morfológico de 7×7 erosionaba los bordes delgados del triángulo rojo.  
**Solución:** Kernel **3×3** + reducir `min_area` a 120 px².

### Reto 3: API de HuggingFace Transformers cambió en v5.9

**Problema:** `pipeline("text2text-generation")` arrojaba `KeyError: "Unknown task"`.  
**Solución:** API de bajo nivel directamente:
```python
from transformers import T5ForConditionalGeneration, AutoTokenizer
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
```

### Reto 4: Múltiples cajas superpuestas en el agente

**Problema:** ColorDetector encontraba varios contornos dentro de la misma señal STOP (borde exterior, espacios entre letras), y el ViT clasificaba cada uno → 5-10 cajas superpuestas.  
**Solución:** En `perceive()`, clasificar todas las regiones y devolver **solo la de mayor confianza**:
```python
best = max(classified, key=lambda x: x[2])
return [best] if best[2] >= CONF_MIN else []
```

### Reto 5: Falsos positivos en paredes y fondos

**Problema:** La pared blanca del fondo y manchas de luz eran detectadas como señales.  
**Solución dual:**
- `CONF_MIN = 0.97` — el ViT debe estar al 97% seguro (paredes nunca superan este umbral)
- `MIN_AREA_AGENT = 600 px²` — manchas de luz son mucho más pequeñas que una señal real

### Reto 6: Caracteres `??` en pantalla (tildes y ñ)

**Problema:** OpenCV no soporta UTF-8. Cualquier tilde (á, é, ñ) aparecía como `??`.  
**Solución:** Función `_limpiar()` que reemplaza caracteres no-ASCII antes de dibujar:
```python
def _limpiar(texto):
    tabla = str.maketrans("áéíóúÁÉÍÓÚñÑ¿¡", "aeiouAEIOUnN  ")
    return texto.translate(tabla)
```

### Reto 7: Flan-T5 repetía el prompt en la respuesta

**Problema:** Con el prompt "You are a driving instructor...", T5 generaba: *"Soy un instructor de conducción explicando una señal de tráfico..."* — repitiendo la instrucción.  
**Solución:** Cambiar a formato Q&A directo:
```
"Question: What does '{nombre}' mean?\nContext: {desc}\nAnswer in Spanish:"
```
Esto produce respuestas directas sin artefactos del prompt.

### Reto 8: Video se congelaba mientras Flan-T5 procesaba

**Problema:** Flan-T5 tarda 3-9 segundos en CPU. Ejecutarlo en el hilo principal bloqueaba el video.  
**Solución:** `threading.Thread` — Flan-T5 corre en un hilo de fondo. El hilo principal sigue capturando y mostrando frames normalmente. El panel se actualiza cuando T5 termina.

---

## 13. Resultados

### Clasificador Visual (DeiT-tiny en GTSRB)

| Época | Acc. Entrenamiento | Acc. Validación | Tiempo |
|---|---|---|---|
| 1 | 91.2% | 98.6% | ~25 min |
| 2 | 99.6% | 99.8% | ~25 min |
| **TEST** | — | **99.05%** | — |

Solo 123 de 12,630 señales de prueba son clasificadas incorrectamente.

### Agente Reactivo en tiempo real

| Prueba | Resultado |
|---|---|
| Señal STOP colombiana "PARE" (foto en teléfono) | Detectada: **STOP, 99.7% confianza** |
| Pared blanca del fondo | **No detectada** (por debajo del umbral 97%) |
| Tiempo de generación Flan-T5 | ~8.7 segundos en CPU |
| Video bloqueado durante generación | **No** — hilo de fondo |

### Asistente Conversacional

| Pregunta | Fuente respuesta | Calidad |
|---|---|---|
| "que significa ceda el paso" | Fallback labels.py | Correcta |
| "que significa curva peligrosa" | Fallback labels.py | Correcta |
| "fin de prohibicion de adelantar" | T5 generando | Correcta y concisa |
| "hay senales para ciclistas" | Fallback labels.py | Correcta |

---

## 14. Estructura de Archivos del Proyecto

```
gtsrb/
├── labels.py           # Base de conocimiento: 43 clases (nombres + descripciones)
├── train_vit.py        # Entrenamiento DeiT-tiny en GTSRB (~50 min en CPU)
├── detector.py         # Deteccion por color HSV + YOLOv8 opcional
│                       #   ColorDetector: min_area configurable por modulo
├── app.py              # App visual: webcam/video/imagen (solo etiquetas)
├── assistant.py        # Asistente Q&A con Flan-T5-base
├── agente.py           # *** AGENTE REACTIVO DUAL-TRANSFORMER ***
│                       #   CONF_MIN=0.97, MIN_AREA=600px2, TRIGGER=6 frames
│                       #   DeiT-tiny -> class_id -> Flan-T5 -> explicacion
├── debug_detector.py   # Herramienta de diagnostico del detector HSV
├── requirements.txt    # Dependencias de Python
├── vit_gtsrb.pth       # Pesos del modelo entrenado (DeiT-tiny, 99.05%)
└── data/
    └── gtsrb/
        └── GTSRB/
            ├── Training/    # ~39,209 imagenes de entrenamiento (PPM)
            └── Test/        # ~12,630 imagenes de prueba
```

---

## 15. Instrucciones de Uso

### Instalación

```bash
pip install torch torchvision transformers ultralytics opencv-python Pillow numpy tqdm
```

### Paso 1: Entrenar el clasificador (solo la primera vez, ~50 min)

```bash
python train_vit.py
# Descarga GTSRB automaticamente (~340 MB)
# Guarda vit_gtsrb.pth con 99.05% de precision
```

### Paso 2: Ejecutar el agente reactivo (modulo principal)

```bash
python agente.py                       # Webcam
python agente.py --video archivo.mp4   # Archivo de video
python agente.py --image foto.jpg      # Imagen estatica
```

**Comportamiento del agente:**
- Detecta señales con confianza ≥ 97%
- Requiere que la señal aparezca en 6 frames consecutivos (anti-falsos positivos)
- Lanza Flan-T5 en hilo de fondo al detectar señal nueva (~8-9 seg en CPU)
- Muestra una sola caja sobre la señal más confiable
- No bloquea el video durante la generación de texto

### Paso 3 (opcional): Asistente conversacional

```bash
python assistant.py
# Primera vez: descarga flan-t5-base (~900 MB, queda en cache)
```

### Paso 4 (opcional): App visual básica

```bash
python app.py                          # Webcam
python app.py --image foto.jpg         # Imagen con multiples senales
python app.py --no-detector            # Clasifica centro del frame
```

---

## 16. Conclusiones

Este proyecto integra múltiples áreas de la Inteligencia Artificial moderna:

1. **Visión por Computador**: detección de objetos por color en espacio HSV, morfología matemática, Non-Maximum Suppression.

2. **Vision Transformers (ViT)**: demostración de que la arquitectura transformer aplicada a imágenes (DeiT-tiny) alcanza 99.05% de precisión en GTSRB con solo 50 minutos de entrenamiento en CPU, usando transfer learning desde ImageNet.

3. **Text Transformers (Flan-T5)**: uso de un encoder-decoder transformer para generar explicaciones automáticas en lenguaje natural a partir de la salida del clasificador visual — el núcleo del requisito académico.

4. **Agente Reactivo**: implementación del ciclo percibir → razonar → actuar que conecta ambos transformers. El agente actúa de forma autónoma al detectar una señal, sin intervención del usuario, usando threading para no bloquear el video.

5. **Ingeniería de robustez**: ajuste iterativo de umbrales (CONF_MIN=0.97, MIN_AREA=600 px²), filtrado a una sola detección, normalización de texto para OpenCV, y prompt engineering para evitar que T5 repita sus instrucciones.

El sistema funciona completamente en CPU (sin GPU), detecta señales colombianas y alemanas, y fue validado en tiempo real con la señal STOP colombiana "PARE" a 99.7% de confianza.

---

*Informe — Mayo 2026 — Andres Felipe Bolaños Zuñiga*
