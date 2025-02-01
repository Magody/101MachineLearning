
from lib.framework_genai.GenAILLM import EnumGenAIModelsIdsOpenAI, EnumGenAIPlatforms, GenAILLM
from lib.framework_genai.GenAIMemory import EnumMemoryType, GenAIMemory
from lib.framework_genai.GenAIPrompt import GenAIPrompt
from lib.framework_genai.LoggingAndTelemetry import EnumLogs
import os
from dotenv import load_dotenv
load_dotenv()

# conda run -n env_ai --no-capture-output python -m lib.framework_genai.tests.test_GenAILLM_prompts

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

chain_resumen = gen_ai_llm.create_chain(
    GenAIPrompt(
        """Realiza un resumen por cada persona involucrada en el chat. El chat es el siguiente:
        {chat}
        Recuerda ser conciso y extraer solo los puntos más importantes, por ejemplo si el chat es:
        '''
        - Nestor: Jajaja chucha ya nada amigo
        - Willy: V: o métete a una pirámide que vende esa webada
        - Willy: Pero luego te vas a sentir mal estafando gente
        - Nestor: Yo estoy en el all World de vanguard y el sp 500 de blackrock y en bitcoins un poquito
        - Nestor: De aquí me descuido de eso un año a ver
        - Nestor: Jajaja cierto
        - Nestor: Voy a publicar en mi tiktok:v
        - Nestor: Cuál es el qqq amigo?
        - Willy: Compra el que compra la gente rica como elon musk
        - Willy: O lo grandes de btx
        '''
        Entonces tu respuesta debe ser como lo siguiente. No es necesario repetir el nombre de la persona que escribe en cada línea. Solo has un resumen:
        '''
        Nestor menciona que:
        1. ...
        2. ...

        Willy menciona que:
        1. ...
        2. ...
        '''
        """,
        partials={}
    )
)

print(chain_resumen.invoke({
    "chat": """- Nestor: Ya sé amigo, cuando se registran con el link puedes escoger en qué cripto quieres tu recompensa
- Nestor: Jajaja chucha ya nada amigo
- Willy: V: o métete a una pirámide que vende esa webada
- Willy: Pero luego te vas a sentir mal estafando gente
- Nestor: Yo estoy en el all World de vanguard y el sp 500 de blackrock y en bitcoins un poquito
- Nestor: De aquí me descuido de eso un año a ver
- Nestor: Jajaja cierto
- Nestor: Voy a publicar en mi tiktok:v
- Nestor: Cuál es el qqq amigo?
- Willy: Compra el que compra la gente rica como elon musk
- Willy: O lo grandes de btx
- Willy: Jaja esos manes compraron Shiba cuando valia 0.0000000001 centavo de dolae
- Willy: Y como a la semana subió a 10 centavos
- Willy: Y luego quedó en 1
- Willy: Mejor creo que te sale ahorrar y comprar ropa y luego vender v':
- Willy: O zapatos
- Willy: O traer cosas de eeuu
- Dogo: El índice NASDAQ 100 amigo
- Dogo: Es bueno porque es estable
- Nestor: Sisi es como el allworld
- Nestor: Pero imagino que el NASDAQ es de vanguard no?
- Dogo: Yo confío en las predicciones del Donno amigo
- Dogo: Deley amigo
- Nestor: Ellas no pero el sistema financiero de EEUU capaz si jaja
- Nestor: Por eso el NASDAQ o el all Word son buenazos buena elección amigo
- Nestor: Creemos una cripto que se llame wololipo
"""
})["output"])