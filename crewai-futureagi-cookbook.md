Cookbook: Build an AI Trading Agent with CrewAI & Future AGI

📖 Our Project Story
In this project, we built a smart cryptocurrency trading assistant that doesn't just answer questions - it automatically checks its own work!

We created a CrewAI agent that can analyze trading strategies and explain investment decisions, then integrated Future AGI to ensure every response is professional, helpful, and safe. The agent can retrieve real trade history data, explain why certain coins were bought/sold, and automatically evaluate the quality of its own advice.

🎯 What We Built
🤖 AI Trading Analyst: CrewAI agent that understands trading questions and provides data-driven insights

🔍 Context-Aware: Integrates with historical trade data using LlamaIndex

✅ Auto-Quality Check: Future AGI evaluates every response for tone, helpfulness, and safety

🛡️ Safety Guardrails: Blocks harmful or toxic queries before processing

📋 Prerequisites
bash
pip install crewai llama-index python-dotenv pandas groq fi-evals
Python 3.10+

Groq API Key

Future AGI API Keys

Trade data in CSV format

🍳 Step-by-Step Recipe
1. Setup Environment
python
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from fi.evals import Evaluator

load_dotenv()

# Initialize Future AGI evaluator
evaluator = Evaluator(
    fi_api_key=os.getenv("FI_API_KEY"),
    fi_secret_key=os.getenv("FI_SECRET_KEY")
)
2. Create Trading Tools
python
from crewai.tools import tool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

@tool("Trade Context Retriever")
def retrieve_trade_context(query: str) -> str:
    """Retrieve relevant trade history for questions"""
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    return "\n".join([n.text for n in nodes[:2]])
3. Build Your Trading Agent
python
# Create LLM connection
groq_llm = LLM(
    model="groq/deepseek-r1-distill-llama-70b",
    temperature=0.7
)

# Define the trading expert
trader_agent = Agent(
    role="Cryptocurrency Trader",
    goal="Provide accurate, helpful trading insights",
    backstory="Experienced crypto trader using data-driven analysis",
    tools=[retrieve_trade_context],
    llm=groq_llm,
    verbose=True
)
4. Add Future AGI Quality Control
python
def evaluate_agent_performance(query: str, response_text: str):
    """Comprehensive quality evaluation"""
    evaluation_results = {}
    
    # Check tone (professional, friendly, etc.)
    tone_result = evaluator.evaluate(
        eval_templates="tone",
        inputs={"input": response_text},
        model_name="turing_flash"
    )
    
    # Check helpfulness
    helpfulness_result = evaluator.evaluate(
        eval_templates="prompt_instruction_adherence",
        inputs={"input": query, "output": response_text},
        model_name="turing_flash"
    )
    
    return evaluation_results
5. Safety Guardrails
python
def check_query_safety(query: str):
    """Block harmful queries before processing"""
    safety_result = evaluator.evaluate(
        eval_templates="toxicity",
        inputs={"input": query},
        model_name="turing_flash"
    )
    # Returns True if safe, False if toxic
    return toxicity_score < 0.5
6. Complete Workflow
python
def run_trading_agent(query: str):
    if not check_query_safety(query):
        return "Sorry, I can't answer that question."
    
    # Execute CrewAI agent
    crew = Crew(agents=[trader_agent], tasks=[trading_task])
    result = crew.kickoff(inputs={"query": query})
    
    # Evaluate response quality
    evaluation = evaluate_agent_performance(query, result.raw)
    
    return result, evaluation
🧪 Example Results
Input: "why did you buy doge coin?"

Agent Response:

"The purchase of Doge Coin was driven by its strong community support and meme culture potential. Historical data showed positive momentum around social media events and influencer endorsements."

Quality Evaluation:

json
{
  "evaluations": {
    "tone": {"score": ["neutral", "professional"]},
    "helpfulness": {"score": 1.0},
    "toxicity": {"score": "Passed"},
    "context_relevance": {"score": 0.92}
  }
}
🚀 Key Features
📊 Data-Driven Insights: Uses real trade history for accurate responses

✅ Automatic Quality Control: Every response evaluated before delivery

🛡️ Content Safety: Blocks harmful or inappropriate queries

📈 Performance Metrics: Quantitative scores for continuous improvement

💡 Business Value
This integration helps:

Financial institutions ensure AI trading advice is accurate and compliant

Crypto platforms provide safe, helpful customer support

Trading firms maintain professional tone in client communications

Developers catch issues before they reach production

🎯 Getting Started
Clone the repository

Add your API keys to .env file

Add trade data in data/trades.csv

Run the agent: python trader_agent.py

Ask questions: "why did you buy ETH?", "should I invest in Bitcoin?"



🗂️ File Structure
text
trader_agent/
├── trader_agent.py          # Main agent code
├── cookbook.md             # This document
├── data/
│   ├── trades.csv          # Trade history data
│   └── documents/          # Additional context files
├── .env                    # API keys (gitignore)
└── requirements.txt        # Dependencies
This cookbook tells the complete story of my project and provides everything another developer needs to replicate your success! 🎉

Social Media summary

⚡ Sometimes the best way to grow is to revisit your old work and level it up.

A few months back, I built a simple AI Trading Agent that analyzed past crypto trades and explained decisions. Recently, I pulled that project back off the shelf — but this time, I added Future AGI’s evaluation layer on top.

🚀 The result?
An agent that doesn’t just answer but also checks the quality and safety of its own advice before giving it to the user.

💡 What’s new:
✅ Quality Evaluation – Every response gets scored for tone, helpfulness, and context relevance
✅ Safety Guardrails – Toxic or harmful queries are blocked early
✅ Data-Driven Insights – Still grounded in real trade history (BTC, DOGE, ETH, etc.)
✅ Future-Ready Design – Built with CrewAI + LlamaIndex, evaluated with Future AGI

This journey reminded me of something my mentor Praveen Sir always emphasizes:
👉 Don’t just build AI that “works” — build AI that is safe, reliable, and production-ready.

Really grateful for the push, because now this project isn’t just a “trading bot” — it’s a step toward trustworthy Agentic AI for FinTech.

#AI #FinTech #FutureAGI #AgenticAI #CrewAI #CryptoTrading
