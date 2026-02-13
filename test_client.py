# MCP Client – MLflow Prediction (minimal)

import asyncio
from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.sse import sse_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()

# URL SSE de ton MCP déployé sur Heroku
MCP_URL = "https://mcp-projet-final-3e91217ab57f.herokuapp.com/sse"

# URL locale d’Ollama
# OLLAMA_BASE_URL = "https://ideal-space-giggle-jwwxppqggxg3pxrj-11434.app.github.dev"
OLLAMA_BASE_URL = "http://localhost:11434"



async def main():
    # Connexion SSE au MCP distant
    async with sse_client(url=MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialisation de la session MCP
            await session.initialize()

            # Chargement des tools exposés par le MCP (ici: predict)
            tools = await load_mcp_tools(session)
            print(f"Loaded tools: {[t.name for t in tools]}")

            # LLM Ollama local (doit supporter le tool calling)
            llm = ChatOllama(
                model="qwen2.5:0.5b:latest",
                base_url=OLLAMA_BASE_URL,
                temperature=0,
            )

            # Création de l’agent avec le tool MCP
            agent = create_agent(llm, tools)

            # Prompt simple pour déclencher le tool predict
            user_prompt = """
Call the predict tool with the following input:
{
  "feature1": 1.2,
  "feature2": 3.4,
  "feature3": "A"
}
Return only the prediction result.
""".strip()

            # Exécution
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=user_prompt)]
            })

            return result


if __name__ == "__main__":
    result = asyncio.run(main())
    final_answer = result["messages"][-1].content
    print("Prediction result:")
    print(final_answer)
