---
title: Reconocimiento Señales Transito
emoji: 🚦
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: gradio_app.py
pinned: false
license: mit
---

# 🚦 Reconocimiento de Señales de Tránsito

Sistema de visión por computador con agente dual-transformer para clasificar
las 43 clases del dataset GTSRB.

- **Transformer 1**: DeiT-tiny (Vision Transformer) — clasifica señales
- **Transformer 2**: Flan-T5-base — genera explicaciones en español
- **Precisión**: 99.05% en GTSRB (12,630 imágenes de prueba)

El enunciado original del challenge académico está en
[`CHALLENGE.md`](CHALLENGE.md).
