
from lib.framework_genai.GenAILLM import EnumGenAIModelsIdsOpenAI, EnumGenAIPlatforms, GenAILLM
from lib.framework_genai.GenAIMemory import EnumMemoryType, GenAIMemory
from lib.framework_genai.LoggingAndTelemetry import EnumLogs
import os
from dotenv import load_dotenv
load_dotenv()

ai_prefix = "Human"
human_prefix = "Assistant"
verbose_level:EnumLogs = EnumLogs.LOG_LEVEL_DEBUG

gen_ai_llm = GenAILLM(
    platform = EnumGenAIPlatforms.PLATFORM_OPENAI,
    model_id = EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_3_5_TURBO,
    parameters_inference = {
        'max_tokens': 1024, 
        "temperature": 0.1,
        "top_p": 0.2
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

print(gen_ai_llm.invoke("Hola cómo estas hoy?"))