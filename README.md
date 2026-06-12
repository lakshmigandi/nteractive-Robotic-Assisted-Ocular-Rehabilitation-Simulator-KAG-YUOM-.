# Interactive Robotic-Assisted Ocular Rehabilitation Simulator (KAG + YUOM)

An advanced medical-engineering agent framework built by **Dr. Lakshmi Gandi**. This platform uses Knowledge-Augmented Generation (KAG) paired with a Unified Operational Matrix (YUOM) to simulate and monitor robotic-assisted neuro-visual rehabilitation workflows.

---

## 🏗️ Architecture Overview

The platform uses a hybrid architectural approach designed to balance speed, factual containment, and conversational capabilities:

* **Knowledge-Augmented Generation (KAG)**: Parses the operational manual (`Ocular_Rehab_Simulator.pdf`) into text matrices using a FAISS vector store running Maximal Marginal Relevance (MMR) retrieval.
* **Deterministic Intent Router**: An inline structural processing layer that instantly handles localized SQLite database resource queries or basic routing commands, mitigating LLM API latency.
* **State Machine Coordination**: Uses LangGraph to orchestrate message payloads and process tool integrations conditionally.
* **Conversational Interface**: Leverages an open-source `Meta-Llama-3-8B-Instruct` model wrapped dynamically via ChatHuggingFace to satisfy conversational endpoint pipelines.

---

## 📦 Project Directory Structure

```text
├── app.py                     # Unified production script (Vector database, SQL backend, LangGraph, and Gradio UI)
├── requirements.txt           # Environment library installation list
├── Ocular_Rehab_Simulator.pdf # Underlying technical engineering and clinical manual
└── README.md                  

