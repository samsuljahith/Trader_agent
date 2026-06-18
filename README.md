# Trader Agent — CrewAI Crypto Assistant

A conversational cryptocurrency trading agent built with CrewAI and LlamaIndex. Retrieves context from historical trade data via RAG and answers strategy questions through an interactive CLI.

## How It Works

1. Loads `data/trades.csv` and indexes it with LlamaIndex + HuggingFace embeddings
2. Detects the trader's persona (Meme Trader / Technical Trader) and risk level from the CSV
3. On each user query, the CrewAI agent retrieves relevant trade context via RAG before responding
4. FutureAGI evaluates every response for tone, helpfulness, context relevance, and toxicity

## Tech Stack

- **Agent framework** — CrewAI
- **LLM** — Groq (`deepseek-r1-distill-llama-70b`)
- **RAG** — LlamaIndex + HuggingFace `BAAI/bge-small-en-v1.5` embeddings
- **Evaluation** — FutureAGI (`fi.evals`)

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
FI_API_KEY=your_futureagi_key
FI_SECRET_KEY=your_futureagi_secret
```

```bash
python trader_agent.py
# Type 'quit' to exit
```

## Data

Place your trade history in `data/trades.csv`. Expected columns include `Tags` and `Outcome` (used for persona detection).
