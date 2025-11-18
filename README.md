# LLM Eval Pro

Evaluate and benchmark Large Language Models (LLMs) for factuality, hallucination, and retrieval accuracy.

##  Overview
**LLM Eval Pro** provides a modular framework for comparing and validating LLM performance across different models and datasets.  
It supports custom evaluation metrics, visual dashboards, and automated reports.

###  Features
- Compare multiple LLMs (OpenAI, Claude, Gemini, etc.)
- Measure hallucination and retrieval accuracy
- Track metrics such as BLEU, BERTScore, recall, and precision
- Visualize results with Matplotlib or W&B dashboards
- Plug-and-play architecture for RAG and agent pipelines

### ⚙️ Tech Stack
- **Languages:** Python, SQL  
- **Frameworks:** LangChain, Pandas, Matplotlib, W&B  
- **Infra:** Docker (optional)  


###  Project Structure

```
llm-eval-pro/
├── data/
│ └── samples.json
│
├── eval/
│ ├── metrics.py
│ ├── compare_models.py
│ └── visualize.py
│
├── configs/
│ └── eval_config.yaml
│
├── notebooks/
│ └── demo.ipynb
│
├── README.md
└── requirements.txt

```
How to run this end-to-end
# 1. Create virtualenv and install deps
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Run evaluation (dummy model)
python -m eval.compare_models --config configs/eval_config.yaml

# 3. Generate charts
python -m eval.visualize
