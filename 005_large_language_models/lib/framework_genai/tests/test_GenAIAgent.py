
from lib.framework_genai.GenAIAgent import EnumGenAIAgentType, GenAIAgent
from lib.framework_genai.GenAILLM import EnumGenAIModelsIdsOpenAI, EnumGenAIPlatforms, GenAILLM
from lib.framework_genai.GenAIMemory import EnumMemoryType, GenAIMemory
from lib.framework_genai.GenAIOutputParser import EnumCommonParsers, GenAIOutputParser
from lib.framework_genai.LoggingAndTelemetry import EnumLogs
import os
from langchain.tools import StructuredTool
import asyncio
from dotenv import load_dotenv
load_dotenv()

# conda run -n env_ai --no-capture-output python -m lib.framework_genai.tests.test_GenAIAgent

ai_prefix = "Human"
human_prefix = "Assistant"
verbose_level:EnumLogs = EnumLogs.LOG_LEVEL_DEBUG

gen_ai_llm = GenAILLM(
    platform = EnumGenAIPlatforms.PLATFORM_OPENAI,
    model_id = EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO,
    parameters_inference = {
        'max_tokens': 1024, 
        "temperature": 0,
        "top_p": 0.1
    },
    ai_prefix = ai_prefix,
    human_prefix = human_prefix,
    component_memory=GenAIMemory(EnumMemoryType.MEMORY_CHAT_BUFFER, ai_prefix, human_prefix),
    platform_configuration = {
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]
    },
    verify_ssl=True,
    verbose_level=verbose_level
)

gen_ai_output_parser = GenAIOutputParser.create_common_parser(EnumCommonParsers.PARSER_REASONING_ANSWER)


async def get_meaning_life(input=""):
    return "42"

tool_meaning_life = StructuredTool.from_function(
    func=get_meaning_life,
    coroutine=get_meaning_life,
    name="MeaningLife",
    description="""Función que indica el sentido de la vida""",
    return_direct=False,
)

tools = [tool_meaning_life]

agent = GenAIAgent(
    system_prompt="""Responde siempre en el formato json indicado""",
    agent_type=EnumGenAIAgentType.AGENT_TYPE_REACT,
    gen_ai_output_parser=gen_ai_output_parser,
    gen_ai_llm=gen_ai_llm,
    tools=tools,
    state={},
    name="TestGPT",
    verbose_level=verbose_level
)

# print(agent.gen_ai_prompt.prompt_template)

async def get_invoke_async():
    result = await agent.ainvoke({
        "input": "Hola, dime cuál es la forma de usar un ciclo for en python",
        # ¿Cuál es el sentido de la vida?
        # "Hola, dime cuál es la forma de usar un ciclo for en python
        "context": ""
    })
    print("resultresult:", result, type(result))
    # print(gen_ai_output_parser.parse(result["output"]))

asyncio.run(get_invoke_async())

# output = agent.invoke({
#     "input": "¿Cuál es el sentido de la vida?",
#     "context": ""
# })
# print(output)
# print(gen_ai_output_parser.parse(output["output"]))