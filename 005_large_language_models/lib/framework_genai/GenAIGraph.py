from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from .GenAIAgent import GenAIAgent

class GenAIGraph:

    def __init__(self, agent_state:TypedDict) -> None:
        self.work_flow = StateGraph(agent_state)

    def build_system(self):
        self.system = self.work_flow.compile()

    @staticmethod
    def create_agent_node(shared_state:TypedDict, agent:GenAIAgent):
        shared_state["input"] = shared_state["messages"][-1].content

        result = agent.invoke(shared_state)
        output_extracted = agent.gen_ai_output_parser.parse(result["output"])
        return {"messages": [HumanMessage(content=output_extracted["respuesta"], name=agent.name)]}