"""
assistant.py — Asistente de senales de transito usando Flan-T5.

Arquitectura hibrida:
  1. Recuperacion por palabras clave  ->  encuentra la senal relevante
  2. Flan-T5 (transformer de texto)   ->  genera respuesta en lenguaje natural

Modelo : google/flan-t5-small (77M parametros, ~300 MB, rapido en CPU)

Uso:
    python assistant.py
"""

import re
import torch
import warnings
warnings.filterwarnings("ignore")
from transformers import T5ForConditionalGeneration, AutoTokenizer, logging
logging.set_verbosity_error()
from labels import CLASS_NAMES, DESCRIPTIONS


# ─── Sinonimos y palabras clave en espanol ────────────────────────────────────

KEYWORDS = {
    # Senales de velocidad
    "velocidad": [0,1,2,3,4,5,6,7,8],
    "limite":    [0,1,2,3,4,5,6,7,8],
    "20":  [0],  "30": [1],  "50": [2],  "60": [3],
    "70":  [4],  "80": [5],  "100":[7],  "120":[8],

    # Senales de prohibicion
    "adelantar": [9, 10, 41, 42],
    "adelantamiento": [9, 10],
    "prohibido": [9, 10, 15, 16, 17],
    "paso":      [13, 15],
    "entrar":    [17],
    "entrada":   [17],
    "contravía": [17],
    "contravia": [17],

    # Prioridad
    "prioridad":    [11, 12],
    "principal":    [12],
    "interseccion": [11],

    # Ceda / Stop
    "ceda":    [13],
    "yield":   [13],
    "stop":    [14],
    "alto":    [14],
    "pare":    [14],
    "detener": [14],
    "detencion":[14],
    "parar":   [14],

    # Vehiculos
    "vehiculo": [15, 16],
    "camion":   [10, 16],
    "toneladas":[10, 16],
    "3.5":      [10, 16, 42],

    # Peligros y curvas
    "precaucion": [18],
    "peligro":    [18, 19, 20, 21, 22, 23, 30, 31],
    "curva":      [19, 20, 21],
    "izquierda":  [19, 34, 37, 39],
    "derecha":    [20, 33, 36, 38],
    "doble":      [21],

    # Estado del pavimento
    "pavimento":    [22, 23],
    "irregular":    [22],
    "deslizante":   [23],
    "resbaladizo":  [23],
    "hielo":        [30],
    "nieve":        [30],

    # Obras y senales de situacion
    "estrecha":  [24],
    "obras":     [25],
    "trabajo":   [25],
    "construccion": [25],
    "semaforo":  [26],
    "luz":       [26],

    # Personas
    "peatones":  [27],
    "peatonal":  [27],
    "ninos":     [28],
    "niños":     [28],
    "escolar":   [28],
    "infantes":  [28],
    "ciclistas": [29],
    "bicicleta": [29],
    "bici":      [29],

    # Animales
    "animales":  [31],
    "ciervos":   [31],
    "fauna":     [31],

    # Direcciones
    "giro":      [33, 34],
    "girar":     [33, 34],
    "doblar":    [33, 34],
    "recto":     [35, 36, 37],
    "seguir":    [35, 36, 37],
    "mantener":  [38, 39],
    "rotonda":   [40],
    "redonda":   [40],
    "glorieta":  [40],

    # Fin de restricciones
    "fin":       [6, 32, 41, 42],
    "termina":   [6, 32, 41, 42],
}


# ─── Asistente ────────────────────────────────────────────────────────────────

class TrafficAssistant:
    """
    Asistente que combina recuperacion por palabras clave + Flan-T5.

    Flujo:
      1. El usuario escribe una pregunta
      2. Se buscan las senales relevantes por palabras clave
      3. Se construye un prompt corto con solo esa informacion
      4. Flan-T5 genera la respuesta en lenguaje natural
    """

    MODEL = "google/flan-t5-base"   # 250M params, mejor calidad en espanol (~900 MB)

    def __init__(self):
        print(f"Cargando Flan-T5 ({self.MODEL})...")
        print("(Primera vez: descarga ~300 MB, luego queda en cache)")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model     = T5ForConditionalGeneration.from_pretrained(self.MODEL)
        self.model.eval()
        self.detected_signs = []
        print("Asistente listo.\n")

    def set_detected_signs(self, signs: list):
        self.detected_signs = signs

    # ── Recuperacion por palabras clave ───────────────────────────────────────

    def _find_relevant_signs(self, question: str) -> list[int]:
        """
        Busca las clases de senales mas relevantes para la pregunta.
        Retorna lista de class_ids ordenados por relevancia.
        """
        q = question.lower()
        # Quitar acentos basicos para comparacion robusta
        q = q.replace("á","a").replace("é","e").replace("í","i") \
             .replace("ó","o").replace("ú","u").replace("ñ","n")

        scores = {}
        for keyword, ids in KEYWORDS.items():
            if keyword in q:
                for cid in ids:
                    scores[cid] = scores.get(cid, 0) + 1

        # Tambien buscar por nombre de senal directamente
        for cid, name in CLASS_NAMES.items():
            name_norm = name.lower().replace("á","a").replace("é","e") \
                            .replace("í","i").replace("ó","o").replace("ú","u")
            words = name_norm.split()
            hits  = sum(1 for w in words if len(w) > 3 and w in q)
            if hits:
                scores[cid] = scores.get(cid, 0) + hits * 2

        # Si hay senales detectadas recientemente, darles prioridad
        for cid, _, _ in self.detected_signs:
            if cid in scores:
                scores[cid] += 3
            else:
                scores[cid] = 1

        return sorted(scores, key=lambda x: -scores[x])[:3]

    # ── Construccion del prompt ───────────────────────────────────────────────

    def _build_prompt(self, question: str, relevant_ids: list[int]) -> str:
        """
        Prompt optimizado para flan-t5-base.
        El modelo entiende instrucciones directas en ingles
        pero responde bien en espanol cuando el contexto esta en espanol.
        """
        if relevant_ids:
            cid     = relevant_ids[0]
            nombre  = CLASS_NAMES[cid]
            desc    = DESCRIPTIONS[cid]
            context = f"Senal: {nombre}. Significado: {desc}"
            if len(relevant_ids) > 1:
                cid2    = relevant_ids[1]
                context += f" | Senal relacionada: {CLASS_NAMES[cid2]}. {DESCRIPTIONS[cid2]}"
        else:
            context = (
                "Las senales rojas prohiben o advierten peligro. "
                "Las amarillas advierten precaucion. "
                "Respetar las senales es obligatorio por ley de transito."
            )

        return (
            f"Answer in Spanish based on this traffic sign information.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    # ── Respuesta ─────────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """
        Responde una pregunta sobre senales de transito.

        Flujo:
          1. Busca la senal relevante por palabras clave
          2. Construye la respuesta base con la descripcion de labels.py
          3. Usa T5 para enriquecer la respuesta si genera algo valido
        """
        relevant_ids = self._find_relevant_signs(question)

        # ── Respuesta base (siempre correcta) ─────────────────────────────────
        if not relevant_ids:
            base = (
                "No encontre una senal especifica para esa pregunta.\n"
                "Prueba mencionando: stop, ceda el paso, velocidad, curva, "
                "peatones, obras, semaforo, etc."
            )
            return base

        cid       = relevant_ids[0]
        nombre    = CLASS_NAMES[cid]
        descripcion = DESCRIPTIONS[cid]
        base      = f"{nombre}\n{descripcion}"

        # ── T5 intenta enriquecer la respuesta ────────────────────────────────
        prompt = self._build_prompt(question, relevant_ids)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                max_length=300, truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        t5_out = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Validar que T5 genero algo coherente
        q_lower    = question.lower()
        t5_lower   = t5_out.lower()
        base_lower = descripcion.lower()

        # Detectar errores de escritura de T5 (mezcla ingles/espanol)
        palabras_garbled = ["zone de", "seal:", "significado:", "senal:", "senales:"]
        tiene_garbling = any(p in t5_lower for p in palabras_garbled)

        t5_valido = (
            len(t5_out) > 25
            and q_lower[:12] not in t5_lower       # no repite la pregunta
            and t5_lower[:20] not in base_lower    # no duplica la descripcion
            and not t5_lower.startswith("tambien") # no empieza con artefacto
            and not t5_lower.startswith("|")       # no empieza con separador
            and not tiene_garbling                 # no tiene texto corrupto de T5
        )

        if t5_valido:
            # T5 genero una buena respuesta, mostrarla como respuesta principal
            return f"{t5_out}"
        else:
            # Fallback: descripcion directa (siempre correcta)
            return f"{nombre}: {descripcion}"


# ─── Chat en terminal ─────────────────────────────────────────────────────────

def chat_loop(bot: TrafficAssistant):
    print("=" * 60)
    print("  Asistente de Senales de Transito  (Flan-T5)")
    print("=" * 60)
    print("Pregunta lo que quieras sobre senales de transito.")
    print("Ejemplos:")
    print("  - que significa ceda el paso")
    print("  - que pasa si me paso un stop")
    print("  - que significa el limite de 50")
    print("  - que hago con una senal de curva peligrosa")
    print("Escribe 'salir' para terminar.\n")

    while True:
        try:
            question = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not question:
            continue
        if question.lower() in ("salir", "exit", "quit", "q"):
            print("Hasta luego.")
            break

        print("Asistente: pensando...")
        answer = bot.ask(question)
        print(f"Asistente: {answer}\n")


if __name__ == "__main__":
    bot = TrafficAssistant()
    chat_loop(bot)
