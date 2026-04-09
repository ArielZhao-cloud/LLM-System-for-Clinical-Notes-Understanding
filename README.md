# Interactive Oncology Clinical Decision Support System (CDSS)

Welcome to the Interactive Oncology CDSS repository. This project contains a multimodal, AI-assisted clinical workflow designed to augment precision oncology decision-making. 

Powered by the DSPy framework and large language models, this system bridges the gap between complex NCCN clinical guidelines and real-time patient care, featuring a unique dual-view architecture for both healthcare providers and patients.

## Key Features

* **Dual Copilot Architecture:** Seamlessly toggle between a "Doctor Workspace" (professional, jargon-heavy, evidence-based) and a "Patient Portal" (empathetic, accessible, 8th-grade reading level).
* **Deterministic Guardrails:** Implements hardcoded, rules-based safety intercepts (e.g., Creatinine Clearance alerts) prior to LLM invocation to ensure clinical safety.
* **State-Locking EHR Simulation:** Features an interactive Regimen Builder that synchronizes state across provider and patient views upon physician sign-off.
* **Modular RAG Pipeline:** Utilizes DSPy signatures to enforce strict, multi-step clinical reasoning, minimizing hallucination and maximizing guideline adherence.

## Software Requirements & Tech Stack

* **Python Version:** Python 3.11+
* **Core Frameworks:** `streamlit`, `dspy-ai`
* **LLM Provider:** DeepSeek (Configured via API)

## Repository Structure

```text
.
├── app.py                              # (Core) The main Streamlit interactive frontend application
├── modular_rag_pipeline.py             # (Core) DSPy agent orchestration and RAG retrieval logic
├── multi_agent_pipeline.py             # (Core) Underlying LLM configurations and base extractors
├── modular_rag_evaluator.py            # (Core) Evaluation script for ablation studies and metrics
│
├── archive/                            # Early development prototypes, dummy tools, and raw data scripts
├── data/                               # Contains raw MIMIC samples
├── oncology_final_reports_adv_rag.json # Processed MIMIC samples
├── oncology_raw_samples.json           # Extracted raw MIMIC samples
├── requirements.txt                    # Python dependencies
└── LICENSE                             # MIT License
```

## Quick Start Guide

### 1. Environment Setup

Set up a clean Python virtual environment and activate it:

```bash
python3 -m venv medvenv
source medvenv/bin/activate  # On Windows use: medvenv\Scripts\activate
```

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. API Configuration

Create a `.env` file in the root directory of this repository and add your DeepSeek API key:

```text
DEEPSEEK_API_KEY="your_api_key_here"
```

### 3. Launching the Application

To run the interactive Clinical Decision Support System, execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser (typically at `http://localhost:8501`).

## Evaluation & Ablation Studies

For academic and evaluation purposes, the `modular_rag_evaluator.py` script is provided to run automated metrics across the extracted MIMIC-IV clinical datasets. Note: This is separate from the primary `app.py` interface.
