from langchain.prompts.prompt import PromptTemplate
import re
from enum import Enum


class EnumPromptsBase(Enum):

    PROMPT_BASE_REACT_CHAT = """{system_prompt}.
    Tienes acceso a las siguientes herramientas:
    {tools}

    Este es un contexto adicional: <context>{context}</context>

    Utilice el siguiente formato puesto en la etiqueta <formato> para generar tu respuesta final:
    <formato>
    Question: la pregunta de entrada que debes responder
    Thought: siempre debes pensar qué hacer
    Action: la acción a tomar, puede ser una de [{tool_names}]
    Action Input: la entrada de la acción
    Observation: el resultado de la acción
    ... (la secuencia Thought,Action,Action Input y Observation puede repetirse tantas veces como sea necesario)
    Thought: Ahora sé la respuesta final
    Final Answer: la respuesta final a la pregunta de entrada original.
    </formato>

    Cuando sepas y generes 'Final Answer',
    este será un string que estará en el formato de las siguientes instrucciones {format_instructions}

    Ahora inicia. Te pasaré un Question del que deberás continuar con la cadena de pensamientos.
    Question: {input}
    Thought: {agent_scratchpad}
    """


class GenAIPrompt:

    def __init__(self, base_template, partials: dict = {}) -> None:
        """ "(Hola {mundo}. {formatting}", partials={
            'formatting': '1234'
        })
        """
        self.base_template = base_template.replace("<custom-encloser>", "{").replace(
            "</custom-encloser>", "}"
        )
        self.partials = partials
        self.create_simple_prompt_template()

    def extract_variables(self, text):
        return re.findall(r"\{([A-z]+)\}", text)

    def create_simple_prompt_template(self):

        self.input_variables = []

        partials_keys = self.partials.keys()
        for var in self.extract_variables(self.base_template):
            if var not in partials_keys:
                self.input_variables.append(var)

        self.prompt_template = PromptTemplate(
            input_variables=self.input_variables, template=self.base_template
        ).partial(**self.partials)
        return self.prompt_template
