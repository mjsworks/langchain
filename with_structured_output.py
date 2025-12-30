from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

##################################
# task 1: define the list of tools
##################################

#################################################
# How does the LLM know what tool to call?
# it looks at the docstrings. that is the
# description of each tool. and looking at that
# description it decides which tool to call.
#################################################
tools = [TavilySearch()]

##################################
# task 2: define the LLM
##################################
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ToolStrategy(AgentResponse),
    system_prompt=(
        "You are a helpful assistant. Use tools when needed and return sources with URLs."
    ),
)


def main():
    print("Hello from ReAct Agent Depth one")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search 3 job posting in AI/ML engineering in greater sydney area in LinkedIn.",
                }
            ]
        }
    )
    print("Final Result:", result["structured_response"])


if __name__ == "__main__":
    main()
