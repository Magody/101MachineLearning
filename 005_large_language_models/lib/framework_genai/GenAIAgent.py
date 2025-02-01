from enum import Enum
from .GenAIOutputParser import GenAIOutputParser
from .GenAIPrompt import EnumPromptsBase, GenAIPrompt
from langchain.agents import AgentExecutor, create_react_agent
from .GenAILLM import GenAILLM
from .LoggingAndTelemetry import Logging, EnumLogs

class EnumGenAIAgentType(Enum):
    AGENT_TYPE_REACT = "REACT"

class GenAIAgent:

    def __init__(
            self,
            system_prompt:str,
            agent_type:EnumGenAIAgentType,
            gen_ai_output_parser:GenAIOutputParser,
            gen_ai_llm:GenAILLM,
            tools:list,
            name:str,
            state:dict={},
            verbose_level:EnumLogs=EnumLogs.LOG_LEVEL_INFO
    ) -> None:
        
        self.agent_type = agent_type
        self.gen_ai_output_parser = gen_ai_output_parser
        self.gen_ai_llm = gen_ai_llm
        self.tools = tools
        self.name = name
        self.state = state
        self.verbose_level = verbose_level
        self.verbose = (self.verbose_level.value >= EnumLogs.LOG_LEVEL_DEBUG.value)
        
        if agent_type == EnumGenAIAgentType.AGENT_TYPE_REACT:

            self.gen_ai_prompt = GenAIPrompt(
                EnumPromptsBase.PROMPT_BASE_REACT_CHAT.value,
                partials={
                    'format_instructions': self.gen_ai_output_parser.get_instructions(),
                    'system_prompt': system_prompt
                }
            )

            self.agent = create_react_agent(
                gen_ai_llm.llm_model,
                tools, 
                prompt=self.gen_ai_prompt.prompt_template
            )
            self.build_agent_executor()
            
        else:
            raise Exception("NOT IMPLEMENTED")

    def build_agent_executor(self):
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=self.verbose, return_intermediate_steps=(self.verbose))
        return self.agent_executor
    
    def invoke(self, input:dict):
        self.state["last_input"] = input
        self.state["last_invoke"] = self.agent_executor.invoke(input)
        return self.state["last_invoke"]
    
    async def ainvoke(self, input:dict):
        """Asyncronous invoke"""
        self.state["last_input"] = input
        self.state["last_invoke"] = await self.agent_executor.ainvoke(input)

        return self.state["last_invoke"]
