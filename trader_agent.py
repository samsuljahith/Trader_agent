from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from groq import Groq
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


  
EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=EMBED_MODEL)

def analyze_persona():
    trades = pd.read_csv("data/trades.csv")
    return {
        "style": "Meme Trader" if "meme" in trades["Tags"].str.cat() else "Technical Trader",
        "risk": "High" if (trades["Outcome"] == "Loss").mean() > 0.4 else "Medium"
    }


def chat_with_trader():
    persona = analyze_persona()
    print(f"Trader Persona: {persona['style']} (Risk: {persona['risk']})")
    print("Ask about trades or strategy. Type 'quit' to exit.\n")

    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break


        retriever = index.as_retriever()
        trade_context_nodes = retriever.retrieve(query)
        trade_context = "\n".join([n.text for n in trade_context_nodes[:2]])  

        response = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a {persona['style']} cryptocurrency trader with {persona['risk']} risk appetite.
                    Always reference specific trades when possible. Here's relevant context:
                    {trade_context}"""
                },
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        
        print(f"Trader: {response.choices[0].message.content}\n")

if __name__ == "__main__":
    chat_with_trader()
