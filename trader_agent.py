import os
import pandas as pd
import json
import traceback
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from groq import Groq
from fi.evals import Evaluator

# Load environment variables
load_dotenv()

# Initialize models and clients
EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
evaluator = Evaluator(
    fi_api_key=os.getenv("FI_API_KEY"),
    fi_secret_key=os.getenv("FI_SECRET_KEY")
)

# Load LlamaIndex data
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=EMBED_MODEL)

# Tool for persona analysis
def analyze_persona():
    trades = pd.read_csv("data/trades.csv")
    return {
        "style": "Meme Trader" if "meme" in trades["Tags"].str.cat().lower() else "Technical Trader",
        "risk": "High" if (trades["Outcome"] == "Loss").mean() > 0.4 else "Medium"
    }

# Tool for retrieving trade context
@tool("Trade Context Retriever")
def retrieve_trade_context(query: str) -> str:
    """Retrieves relevant trade context for a given query using LlamaIndex."""
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    return "\n".join([n.text for n in nodes[:2]])

# Create Groq LLM for CrewAI
groq_llm = LLM(
    model="groq/deepseek-r1-distill-llama-70b",
    temperature=0.7
)

# Define the CrewAI Agent
trader_agent = Agent(
    role="Cryptocurrency Trader",
    goal="Answer user queries about trading strategies and market insights with accurate, helpful responses",
    backstory="You are an experienced crypto trader who uses data-driven insights and market analysis.",
    tools=[retrieve_trade_context],
    llm=groq_llm,
    verbose=True
)

# Define the Task
trader_task = Task(
    description="Answer the user query: {query}, using trade context and persona analysis.",
    expected_output="A clear, professional response addressing the query with relevant trade data.",
    agent=trader_agent
)

# Safety check with Future AGI - FIXED
def check_query_safety(query: str):
    try:
        safety_result = evaluator.evaluate(
            eval_templates="toxicity",
            inputs={"input": query},
            model_name="turing_flash"
        )
        
        if hasattr(safety_result, 'eval_results') and safety_result.eval_results and len(safety_result.eval_results) > 0:
            output = safety_result.eval_results[0].output
            
            # Handle different output formats
            if isinstance(output, dict):
                # Check for toxicity score in dictionary
                toxicity_score = output.get('toxicity', output.get('score', 0.5))
                is_safe = toxicity_score < 0.5  # Threshold for safe
            elif isinstance(output, (int, float)):
                is_safe = output < 0.5  # Assuming lower values are safer
            else:
                # Default to safe if format is unexpected
                is_safe = True
                
            return is_safe
        else:
            print("Warning: No safety evaluation results. Defaulting to safe.")
            return True
    except Exception as e:
        print(f"Error in safety check: {e}. Defaulting to safe for query: '{query}'.")
        return True

# Run the CrewAI workflow
def run_trader_workflow(query: str):
    persona = analyze_persona()
    crew = Crew(
        agents=[trader_agent],
        tasks=[trader_task],
        verbose=True
    )
    result = crew.kickoff(inputs={"query": query})
    return result

# COMPREHENSIVE EVALUATION FUNCTION - FIXED
def evaluate_agent_performance(query: str, response_text: str):
    """Comprehensive evaluation of agent performance using Future AGI"""
    
    evaluation_results = {}
    
    try:
        # 1. Evaluate Tone
        tone_result = evaluator.evaluate(
            eval_templates="tone",
            inputs={"input": response_text},
            model_name="turing_flash"
        )
        if tone_result.eval_results and len(tone_result.eval_results) > 0:
            tone_data = tone_result.eval_results[0].output
            evaluation_results['tone'] = tone_data
        
        # 2. Evaluate Helpfulness
        helpfulness_result = evaluator.evaluate(
            eval_templates="prompt_instruction_adherence",
            inputs={"input": query, "output": response_text},
            model_name="turing_flash"
        )
        if helpfulness_result.eval_results and len(helpfulness_result.eval_results) > 0:
            helpfulness_data = helpfulness_result.eval_results[0].output
            evaluation_results['helpfulness'] = helpfulness_data
        
        # 3. Evaluate Context Relevance
        # 3. Evaluate Context Relevance - FIXED INPUT FORMAT
        context_result = evaluator.evaluate(
            eval_templates="context_relevance",
            inputs={
                "context": "Cryptocurrency trading analysis",  # Add relevant context
                "input": query
            },
            model_name="turing_flash"
        )
        if context_result.eval_results and len(context_result.eval_results) > 0:
            context_data = context_result.eval_results[0].output
            evaluation_results['context_relevance'] = context_data
        
        # 4. Evaluate Toxicity
        toxicity_result = evaluator.evaluate(
            eval_templates="toxicity",
            inputs={"input": response_text},
            model_name="turing_flash"
        )
        if toxicity_result.eval_results and len(toxicity_result.eval_results) > 0:
            toxicity_data = toxicity_result.eval_results[0].output
            evaluation_results['toxicity'] = toxicity_data
        
        # Print evaluation results
        print("\n" + "="*60)
        print("FUTURE AGI EVALUATION RESULTS:")
        print("="*60)
        
        # Convert to desired format
        formatted_results = {
            "evaluations": {},
            "observability": {"model": "turing_flash"}
        }
        
        for metric, value in evaluation_results.items():
            if isinstance(value, dict):
                formatted_results["evaluations"][metric] = value
            else:
                formatted_results["evaluations"][metric] = {"score": value}
        
        # Pretty print the results
        print(json.dumps(formatted_results, indent=2))
        print("="*60 + "\n")
        
        return formatted_results
        
    except Exception as e:
        print(f"Error in comprehensive evaluation: {e}")
        traceback.print_exc()
        return {}

# Interactive chat loop with evaluation - FIXED INDENTATION
def chat_with_trader():
    persona = analyze_persona()
    print(f"Trader Persona: {persona['style']} (Risk: {persona['risk']})")
    print("Ask about trades or strategy. Type 'quit' to exit.\n")

    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break

        if not check_query_safety(query):
            print("Trader: Sorry, that query seems inappropriate. Try something else.\n")
            continue

        # Get response from agent
        response = run_trader_workflow(query)
        
        # Extract the raw text from CrewOutput object for evaluation
        if hasattr(response, 'raw'):
            response_text = response.raw
        else:
            response_text = str(response)
        
        # Run comprehensive evaluation on the text response
        evaluation_results = evaluate_agent_performance(query, response_text)
        
        # Display the response to the user
        print(f"Trader: {response_text}\n")

if __name__ == "__main__":
    chat_with_trader()