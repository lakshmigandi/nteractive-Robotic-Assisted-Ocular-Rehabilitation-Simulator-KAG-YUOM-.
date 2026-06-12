---
title: Interactive Robotic-Assisted Ocular Rehabilitation Simulator (KAG+YUOM)
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: KAG and YUOM framework for advanced neuro-visual rehabilitation
---

# Interactive Robotic-Assisted Ocular Rehabilitation Simulator (KAG + YUOM)

This repository hosts the live, cloud-native deployment of the **Interactive Robotic-Assisted Ocular Rehabilitation Simulator**, driven by an advanced Knowledge-Augmented Generation (KAG) pipeline and a Unified Operational Matrix (YUOM).

## 🚀 System Components
* **`app.py`**: The single-file production core containing the FAISS vector indexing engine, local SQLite registry, LangGraph state machine, and the Gradio user interface.
* **`requirements.txt`**: The cloud environment dependency blueprint.
* **`Ocular_Rehab_Simulator.pdf`**: The underlying technical patent and engineering manual source matrix.

## 🛠️ Local Setup Tracking
To test this environment layout locally on your desktop machine instead of the cloud container, execute:

```bash
pip install -r requirements.txt
python app.py
