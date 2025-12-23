from dotenv import load_dotenv
load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

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
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
chain = agent_executor

def main():
    print("Hello from ReAct Agent Depth one")
    result = chain.invoke(
        input = {
            "input": ["Search 3 job posting in AI/ML engineering in greater sydney area in LinkedIn."]
        }
    )
    print("Final Result:", result)

if __name__ == "__main__":
    main()