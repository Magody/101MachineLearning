from langchain.prompts.prompt import PromptTemplate
import re
from enum import Enum


class EnumPromptsBase(Enum):
    """
    This class defines a set of prompt templates for various use cases.
    Each prompt template is defined as an Enum member, with the template
    string stored as the value.
    """

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

    Cuando sepas y generes 'Final Answer', este será un string que estará en el formato de las siguientes instrucciones {format_instructions}

    Ahora inicia. Te pasaré un Question del que deberás continuar con la cadena de pensamientos.
    Question: {input}
    Thought: {agent_scratchpad}
    """

    PROMPT_AGENT_REACT_CHAT = """{system_prompt}\n\nTOOLS:\n------\n\nAssistant has access to the following tools:\n\n{tools}\n\n
    To use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nWhen you need information from the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I have a response or question to say to the Human? Yes\nFinal Answer: [your question here]\n```\n
    Begin!\n\nPrevious conversation history:\n{chat_history}\n\nNew user_input: {user_input}\n{agent_scratchpad}
    """

    PROMPT_TEMPLATE_AGENT_REACT_BY_CHAT_OPTIONS = """Answer the following questions and obey the following commands as best you can.

    You have access to the following tools:
    {tools_definition}

    You will receive a message from the human, then you should start a loop and do one of two things

    Option 1: You use a tool to answer the question.
    For this, you should use the following format:
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tools_names}]
    Action Input: "the input to the action, to be sent to the tool"

    After this, the tool will respond with an observation, and you will continue.

    Option 2: You respond to the human.
    For this, you should use the following format:
    Action: {action_human}
    Action Input: "your response to the human, precise and concise"

    Begin! {system_prompt}

    Previous chat history: {chat_history}

    Human: {user_input}
    Assistant: {agent_scratchpad}
    """

    PROMPT_TEMPLATE_AGENT_REACT_BY_CHAT_OPTIONS_ES = """
    {system_prompt}
    Tienes acceso a las siguientes herramientas:
    {tools_definition}

    Recibirás un mensaje del humano, entonces deberás hacer una de estas dos cosas

    Opción 1: Utilizas una herramienta para responder a la pregunta.
    Para ello, debes utilizar el siguiente formato:
    Thought: siempre debes pensar qué hacer
    Action: la acción a realizar, debe ser una de [{tools_names}]
    Action Input: "la entrada de la acción, que se enviará a la herramienta"

    Después de esto, la herramienta responderá con una observación, y continuarás.

    Opción 2: Usa esta herramienta cuando necesites información o responder al humano.
    Para ello, debe utilizar el siguiente formato:
    Action: {action_human}
    Action Input: "su respuesta al humano, precisa y concisa"

    ¡Comencemos! Tu respuesta debe estar en español.

    Historial de chat anterior: {chat_history}

    Human: {user_input}
    Assistant: {agent_scratchpad}
    """

    PROMPT_TEMPLATE_AGENT_REACT_TARGET_ANTHROPIC = """
    Human:
    You are a helfull assistant, your task is help the user achieve is goal.
    In this environment you have access to a set of tools you can use to answer the user's question.
    Use them in a smart way to fullfill our objetives. Always consider the history of previous steps to ensure a continuos progress towards the objetive.
    When you have the answer for the user, always answer it in this forma bellow:
    Final Answer: answer to user


    You may call them like this. Only invoke one function at a time and wait for the results before invoking another function:
    <function_calls>
    <invoke>
    <tool_name>$TOOL_NAME</tool_name>
    <parameters>
    <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
    ...
    </parameters>
    </invoke>
    </function_calls>

    Here are the tools available:
    <tools>
    {tools_string}
    </tools>


    User:
    {input}
    """

    PROMPT_TEMPLATE_CODE_DOCUMENTER = """Eres un experto en {programming_language_to_doc}. El siguiente programa estará en {programming_language_to_doc}.
    Agrega documentación de cada función bajo los estándares y clase manteniendo la congruencia y la ejecutabilidad del programa. 
    Conserva el código, solo agrega los doc strings. Es decir, tu tarea no es eliminar texto ni modificarlo. 
    Es conservarlo y agragar información adicional. La documentación debe estar antes de la función o clase. No después. 
    Solo responde con el código migrado sin comillas especiales como ```{programming_language_to_doc} o similar. Va la respuesta directa              
    Realiza la documentación en inglés
    '''
    <code>{code}</code>
    '''
    """

    PROMPT_TEMPLATE_CODE_UNIT_TESTER = """Eres un experto en {programming_language_for_unit_testing}. El siguiente programa estará en {programming_language_for_unit_testing}, debes 
    realizar un código de pruebas unitarias para validar que la funcionalidad es adecuada.
    Incluye los asserts convenientes y útiles.
    Responde directamente con las pruebas unitarias del código administrado, no realices comentarios adicionales
    {extra_prompt}

    '''
    <code>{code}</code>
    '''
    """

    PROMPT_TEMPLATE_CODE_STRUCTURE_ANALIZER = """Necesito tu ayuda para realizar una tarea de migración de código de un lenguaje a otro. Dada la limitación de la ventana de contexto, necesitamos dividir el código en pedazos. Sin embargo, para reducir el riesgo de alucinación, necesitamos que primero comprendas y generes un resumen de la estructura general del código, una especie de 'esqueleto' que sirva como contexto para los análisis posteriores.
    Por favor, toma la siguiente clase de código como entrada en {programming_language_origin} ubicada en la etiqueta '<code>' y genera un resumen detallado que incluya los componentes principales, las funciones, las variables y cualquier otra estructura relevante. Este resumen debe ser lo suficientemente informativo para proporcionar un entendimiento claro del código y servir como base para la migración del código a otro lenguaje.

    Recuerda, el objetivo es entender la lógica y estructura del código para poder replicarlo en otro lenguaje de programación de manera eficiente y precisa. Por favor, proporciona el resumen en un formato claro y fácil de entender.

    Por ejemplo si la clase de entrada estuviese en Python:

    class Student:
        def __init__(self, name, age):
            self.name = name
            self.age = age

        def get_name(self):
            return self.name

        def get_age(self):
            return self.age

        def set_name(self, name):
            self.name = name

        def set_age(self, age):
            self.age = age

    La estructura 'ósea' generada es:

    Clase: Student
    Atributos: 
    - name
    - age

    Métodos:
    - Constructor (__init__): Toma dos parámetros (name, age) y los asigna a los atributos de la clase.
    - get_name: No toma parámetros. Retorna el valor del atributo 'name'.
    - get_age: No toma parámetros. Retorna el valor del atributo 'age'.
    - set_name: Toma un parámetro (name) y lo asigna al atributo 'name'.
    - set_age: Toma un parámetro (age) y lo asigna al atributo 'age'.
    '''
    <code>{code}</code>
    '''
    """

    PROMPT_TEMPLATE_CODE_TRANSCODER = """Eres un experto en {programming_language_origin} y {programming_language_dest} con más de 5 años de experiencia.
    El siguiente programa que el usuario te pasará estará en {programming_language_origin} en la etiqueta '<code>'.
    Transforma el código a {programming_language_dest} manteniendo la congruencia y la ejecutabilidad del programa.
    Solo responde con el código migrado sin comillas especiales como ```{programming_language_dest} o similar. Va la respuesta directa
    {extra_prompt}
    '''
    <code>{code}</code>
    '''
    """




class GenAIPrompt:
    """
    A class that generates AI prompts based on a base template and optional partial values.

    Attributes:
        __base_template (str): The base template for the AI prompt, with placeholders enclosed in custom tags.
        partials (dict): A dictionary of partial values to be substituted in the base template.
        input_variables (list): A list of variables that need to be provided as input to the prompt template.
        prompt_template (PromptTemplate): The final prompt template with partial values substituted.

    Methods:
        extract_variables(text: str) -> list:
            Extracts the variables from the given text, enclosed in curly braces.
        create_simple_prompt_template() -> PromptTemplate:
            Creates a simple prompt template by extracting the input variables and substituting the partial values.
    """

    def __init__(self, base_template, partials: dict = {}) -> None:
        """
        Initializes the GenAIPrompt object with the base template and optional partial values.

        Args:
            base_template (str): The base template for the AI prompt, with placeholders enclosed in custom tags.
            partials (dict, optional): A dictionary of partial values to be substituted in the base template. Defaults to {}.
        """
        self.__base_template = base_template.replace("<custom-encloser>", "{").replace(
            "</custom-encloser>", "}"
        ).strip()
        self.partials = partials
        self.create_simple_prompt_template()

    def extract_variables(self, text):
        """
        Extracts the variables from the given text, enclosed in curly braces.

        Args:
            text (str): The text from which to extract the variables.

        Returns:
            list: A list of the extracted variables.
        """
        return re.findall(r"\{([A-z]+)\}", text)

    def create_simple_prompt_template(self):
        """
        Creates a simple prompt template by extracting the input variables and substituting the partial values.

        Returns:
            PromptTemplate: The final prompt template with partial values substituted.
        """

        self.input_variables = []

        partials_keys = self.partials.keys()
        for var in self.extract_variables(self.__base_template):
            if var not in partials_keys:
                self.input_variables.append(var)

        self.prompt_template = PromptTemplate(
            input_variables=self.input_variables, template=self.__base_template
        ).partial(**self.partials)
        return self.prompt_template
