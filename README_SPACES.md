---
title: Reconocimiento Señales Transito
emoji: 🚦
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_app.py
pinned: false
license: mit
---

# 🚦 Reconocimiento de Señales de Tránsito

Sistema de visión por computador con agente dual-transformer.

- **Transformer 1**: DeiT-tiny (Vision Transformer) — clasifica señales
- **Transformer 2**: Flan-T5-base — genera explicaciones en español
- **Precisión**: 99.05% en GTSRB (12,630 imágenes de prueba)
