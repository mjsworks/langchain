import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient


# tavily = TavilyClient()

# function to perform a web search
# @tool
# def search_tool(query: str)-> str:
#     """
#     tool that searches over the internet

#     Args:
#         query (str): search query
#     Returns:
#         str: search results
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)

# lanchain tavily tool
from langchain_tavily import TavilySearch

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# tools = [search_tool]
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
)

def main():
    print("Intro to LangChain tools and agents")
    # langchain expects a dict with list
    userMessage = HumanMessage(content="find me 3 job openings for AI engineering in greater sydney area")
    messages = {"messages": [userMessage]}
    response = agent.invoke(messages)
    print(response)

if __name__ == "__main__":
    main()