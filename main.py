from dotenv import load_dotenv

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
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
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instruction = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=[
        "input",
        "agent_scratchpad",
        "tools",
        "tool_names",
    ],
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instruction,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

chain = agent_executor | extract_output | parse_output


def main():
    print("Hello from ReAct Agent Depth one")
    result = chain.invoke(
        input={
            "input": [
                "Search 3 job posting in AI/ML engineering in greater sydney area in LinkedIn."
            ]
        }
    )
    print("Final Result:", result)


if __name__ == "__main__":
    main()
