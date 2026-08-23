\# Grounded Answer — Calder County HSP



A grounded question-answering system for the Calder County Household Support Program policy manual.



\## Features



\- BM25 policy retrieval

\- Groq-powered answer generation

\- Answers grounded only in retrieved policy evidence

\- Insufficient-evidence handling

\- Clause references in answers

\- Automated tests



\## Setup



Create and activate a virtual environment:



&#x20;   python -m venv .venv

&#x20;   .venv\\Scripts\\Activate.ps1



Install dependencies:



&#x20;   pip install -r requirements.txt



Create a local `.env` file containing your Groq API key and model.



The `.env` file is ignored by Git and must never be committed.



\## Run



Ask a question:



&#x20;   python src/main.py --question "What is the income threshold for a household of 4?"



Run tests:



&#x20;   python -m pytest -q



\## Grounding Behavior



The system retrieves relevant policy clauses before generating an answer. The language model receives only the retrieved policy evidence.



For questions that cannot be answered from the policy evidence, it returns:



&#x20;   The available policy evidence is insufficient to answer this question.



\## Technology



\- Python

\- BM25

\- Groq API

\- OpenAI-compatible Python client

\- pytest

