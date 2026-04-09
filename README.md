# Interactive Oncology Clinical Decision Support System (CDSS)
Welcome to the Interactive Oncology CDSS repository. This project implements a multimodal AI-assisted clinical workflow to support precision oncology decision-making.

Powered by the **DSPy framework** and large language models, the system connects complex NCCN clinical guidelines with real-time patient care, supported by a dual-view architecture for both clinicians and patients.

---

## Key Features
- **Dual Copilot Architecture**
  Switch seamlessly between:
  - *Doctor Workspace*: Professional, evidence-based, clinical jargon
  - *Patient Portal*: Empathetic, accessible language (8th-grade reading level)

- **Deterministic Clinical Guardrails**
  Rule-based safety checks (e.g., Creatinine Clearance alerts) run before LLM inference to ensure clinical safety.

- **State-Locking EHR Simulation**
  Interactive Regimen Builder with synchronized state between clinician and patient interfaces after physician sign-off.

- **Modular RAG Pipeline**
  Structured DSPy signatures enforce step-by-step clinical reasoning, reducing hallucinations and improving guideline adherence.

---

## Tech Stack & Requirements
- **Python**: 3.11+
- **Core Frameworks**: `streamlit`, `dspy-ai`
- **LLM**: Zhipu GLM‑4 (API‑based)

---

## Repository Structure
```text
.
├── app.py                              # Main Streamlit interactive frontend
├── modular_rag_pipeline.py             # DSPy agent orchestration & RAG logic
├── multi_agent_pipeline.py             # LLM configuration & base extractors
├── modular_rag_evaluator.py            # Evaluation & ablation study metrics
│
├── archive/                            # Early prototypes & dev scripts
├── data/                               # Raw MIMIC sample datasets
├── oncology_final_reports_adv_rag.json # Processed MIMIC reports
├── oncology_raw_samples.json           # Extracted raw MIMIC samples
├── requirements.txt                    # Python dependencies
└── LICENSE                             # MIT License
```

---

## Quick Start Guide

### 1. Environment Setup
Create and activate a virtual environment:
```bash
python3 -m venv medvenv
source medvenv/bin/activate  # Windows: medvenv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. API Configuration
Create a `.env` file in the root directory and add your Zhipu API key:
```text
ZHIPU_API_KEY="your_api_key_here"
```

### 3. Run the Application
```bash
streamlit run app.py
```
The system will launch in your browser at `http://localhost:8501`.

---

## Evaluation & Ablation Studies
`modular_rag_evaluator.py` supports automated metric evaluation on MIMIC‑IV datasets.
This module is independent of the main `app.py` frontend interface.