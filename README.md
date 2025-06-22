# kp-llm

This minimal project fine-tunes GPT-2 on the Wikitext-2 dataset and exposes a FastAPI service to generate text. Sentry is used for error logging and performance tracing.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file based on the provided template:
   ```env
   SENTRY_DSN=<your dsn>
   ENVIRONMENT=development
   HF_MODEL_NAME=gpt2
   ```
3. Train the model (optional, will use pre-trained GPT-2 if not run):
   ```bash
   python model.py
   ```
4. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```
