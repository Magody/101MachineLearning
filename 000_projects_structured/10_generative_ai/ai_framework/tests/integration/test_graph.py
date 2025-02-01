import httpx
from ai_framework.GenAIGraph import GenAIGraph
from ai_framework.GenAILLM import (
    EnumGenAIModelsIdsBedrock,
    EnumGenAIPlatforms,
    GenAILLM,
)
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAINode import GenAINodeCustom
from ai_framework.GenAINodeChain import EnumGenAINodeChainType, GenAINodeChain
from ai_framework.GenAIPrompt import GenAIPrompt
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging
from ai_framework.test.utils import get_mode, get_file_in_framework
import os
import sys
import pandas as pd
import random
from typing import Union
from langchain.tools import BaseTool
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from typing import TypedDict, Annotated, Any
import operator
import ssl
import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning

ssl._create_default_https_context = ssl._create_unverified_context

mode = get_mode(sys.argv)


def run_graph():
    pass


def test_run_graph():
    if mode in ("test", "prod"):

        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_DEBUG
        verify_ssl = False

        gen_ai_memory = GenAIMemory()

        gen_ai_llm_4k = GenAILLM(
            platform=EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK,  # EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK EnumGenAIPlatforms.PLATFORM_OPENAI
            model_id=EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V2_1,  # EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V2_1, EnumGenAIModelsIdsOpenAI.MODEL_CHAT_GPT_4
            parameters_inference={
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 0,
                "stop_sequences": ["User"],
                "max_tokens_to_sample": 1024,
                "top_k": 5,
            },
            platform_configuration={
                # "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                "aws_access_key_id": os.environ["aws_access_key_id"],
                "aws_secret_access_key": os.environ["aws_secret_access_key"],
                "region_name": os.environ["region_name"],
            },
            verify_ssl=verify_ssl,
            verbose_level=verbose_level,
            metadata=pd.read_csv(get_file_in_framework("test/model_prices.csv"), delimiter=";"),
        )
        """
        configuration_embedder = {
            "provider": {
                "region": os.environ.get("region_name", None),
                "credentials": {
                    "aws_access_key_id": os.environ.get("aws_access_key_id", None),
                    "aws_secret_access_key": os.environ.get(
                        "aws_secret_access_key", None
                    ),
                },
            },
            "model": {"id": "amazon.titan-embed-text-v1"},
            "splitter": {
                "chunk_size": 500,
                "chunk_overlap": 20,
                "length_function": Embedder.num_tokens_from_string,
                "add_start_index": True,
            },
        }

        embedder = EmbedderAWS(configuration_embedder, verify=verify_ssl)
        """

        # Definir listas de datos sintéticos
        identificacion = [random.randint(10000000, 99999999) for _ in range(10)]
        nombre = [
            "Juan",
            "María",
            "Carlos",
            "Laura",
            "Pedro",
            "Ana",
            "Luis",
            "Sofía",
            "Diego",
            "Elena",
        ]
        correo = [
            "juan@example.com",
            "maria@example.com",
            "carlos@example.com",
            "laura@example.com",
            "pedro@example.com",
            "ana@example.com",
            "luis@example.com",
            "sofia@example.com",
            "diego@example.com",
            "elena@example.com",
        ]
        # afiliado_plan_ahorro = [random.choice(["si", "no"]) for _ in range(10)]
        entidad = [random.choice(["Banco", "Asociación"]) for _ in range(10)]
        cedula = [
            random.choice([str(random.randint(10000000, 99999999)), ""])
            for _ in range(10)
        ]
        meses_permanencia = [random.randint(1, 12) for _ in range(10)]
        tipo_contrato = [random.choice(["definido", "indefinido"]) for _ in range(10)]

        # Crear DataFrame
        colaboradores_df = pd.DataFrame(
            {
                "identificacion": identificacion,
                "nombre": nombre,
                "correo": correo,
                # 'afiliado_plan_ahorro': afiliado_plan_ahorro,
                "entidad": entidad,
                "cedula": cedula,
                "meses_permanencia": meses_permanencia,
                "tipo_contrato": tipo_contrato,
            }
        )

        # Actualizar la columna colaborador_nacional en función de si el colaborador tiene una cedula
        colaboradores_df["colaborador_nacional"] = colaboradores_df["cedula"].apply(
            lambda x: "si" if x != "" else "no"
        )

        class PerteneceBancoAsociacionTool(BaseTool):
            name = "Pertenece Banco o Asociación"
            description = "Use this tool to validate if the collaborator belongs to the bank or an association."

            def _run(self, identificacion: Union[int, str]):
                entidad = colaboradores_df[
                    colaboradores_df["identificacion"] == int(identificacion)
                ]["entidad"]
                return entidad.values.tolist() if not entidad.empty else []

            def _arun(self, identificacion: Union[int, str]):
                raise NotImplementedError("This tool does not support async")

        class CumpleRequisitosColaboradorTool(BaseTool):
            name = "Cumple Requisitos de Colaborador"
            description = "Use this tool to validate if the collaborator meets certain requirements."

            def _run(self, identificacion: Union[int, str]):
                colaborador = colaboradores_df[
                    colaboradores_df["identificacion"] == int(identificacion)
                ]
                if not colaborador.empty:
                    meses_permanencia = colaborador["meses_permanencia"].iloc[0]
                    tipo_contrato = colaborador["tipo_contrato"].iloc[0]
                    colaborador_nacional = colaborador["colaborador_nacional"].iloc[0]

                    if meses_permanencia < 1:
                        return (
                            False,
                            "El colaborador no lleva mínimo 1 mes de permanencia",
                        )
                    elif tipo_contrato != "indefinido":
                        return (
                            False,
                            "El tipo de contrato del colaborador no es indefinido",
                        )
                    elif colaborador_nacional != "si":
                        return (False, "El colaborador no es nacional")
                    else:
                        return (True, "El colaborador cumple con todos los requisitos")
                else:
                    return (
                        False,
                        "No se encontró ningún colaborador con la identificación proporcionada",
                    )

            def _arun(self, identificacion: Union[int, str]):
                raise NotImplementedError("This tool does not support async")

        # Crear instancias de herramientas
        pertenece_banco_asociacion_tool = PerteneceBancoAsociacionTool()
        cumple_requisitos_colaborador_tool = CumpleRequisitosColaboradorTool()

        # Lista de herramientas
        tools = [pertenece_banco_asociacion_tool, cumple_requisitos_colaborador_tool]
        tools_names = [
            pertenece_banco_asociacion_tool.name,
            cumple_requisitos_colaborador_tool.name,
        ]

        _REACT_CHAT_PHF_TEMPLATE = """
        Eres un asistente virtual experto en el proceso afiliacion de un colaborador de un banco.\n\n
        Para cumplir con esta tarea debes de seguir los siguientes pasos:
        Primero, debes de saludar y solicitar la identificación del usuario.
        A continuación, con esta identificación debes de validar si el colaborador pertenece al Banco o a una Asociacion.
        En caso de que el colaborador pertenezca a una Asociación debes de comunicarle que no puedes ayudarle en este proceso de afiliacion
        porque no pertenece al banco y despedirte.
        De lo contrario, debes de proceder a validar si el colaborador cumple con los Requisitos de colaborador.
        En caso de que no los cumpla debes de indicarle que requisitos no cumple, indicarle que por ello no podras continuar y despedirte.
        De lo contrario, debes de indicarle que puede acceder a un simulador indicandole Infórmate,
        simula tus aportaciones del plan de ahorro: 
        https://4u.pichincha.com/estatico/plan-futuro-seguro y pidele que despues de la simulacion indique si desea afíliarse de manera simple y guiada.
        En caso de que el usuario se niegue a afiliarse haz esto>(Pregunta al usuario si le gustaria afiliarse mas adelante. Espera su respuesta.
        En caso de una respuesta negativa, despidete amenamente.
        De lo contrario, explicale que un correo, en donde se le recuerde que la afiliacion puede llevarse a acabo del 1 al 10 de cada mes,
        le será enviado el primer lunes del mes actual o, en su defecto, del siguiente mes.), de lo contrario haz esto>(Indicale los pasos a seguir.
        1. Aceptar terminos y condicciones. 2. Completar la documentacion. 3. Entrega fisica de la documentacion. 
        y tambien indicale que la documentacion que debe completar y entregar es la siguiente: 3 copias de contrato impresos a doble cara,
        Formulario Conozca a su cliente impreso a doble cara, Copia de cédula de ambos lados, Papeleta de votación de ambos lados.
        Finalmente indicale que puedo hacerlo a través de https://4u.pichincha.com/estatico/plan-futuro-seguro.) y despidete. TOOLS:\n------
        \n\nAssistant has access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```
        Thought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action\nObservation: the result of the action\n```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        Thought: Do I need to use a tool? No\nFinal Answer: [your response here]\n```

        Begin!\n\nPrevious conversation history:\n{chat_history}\n\nNew human_input: {user_input}\n{agent_scratchpad}
        """

        memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="user_input", return_messages=True
        )

        client = httpx.Client(verify=verify_ssl)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, http_client=client)

        saldos_prompt_react_chat = PromptTemplate(
            input_variables=[
                "agent_scratchpad",
                "chat_history",
                "user_input",
                "tool_names",
                "tools",
            ],
            template=_REACT_CHAT_PHF_TEMPLATE,
        )

        agent = create_react_agent(llm, tools, saldos_prompt_react_chat)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            tools_mames=tools_names,
            memory=memory,
            verbose=True,
        )

        def func_forward_and_save_chain(
            node: GenAINodeChain, shared_state, gen_ai_memory
        ):
            Logging.log(f"RUNNING: {node.id}", node.verbose_level)
            chain_result = node.get_chain_result(shared_state, gen_ai_memory)
            return {
                **shared_state,
                "intermediate_steps_bp": [(node.id, "default")],
                **chain_result,
                "node_outcome": chain_result,
            }

        def func_forward_and_save_rag(
            node: GenAINodeChain, shared_state, gen_ai_memory
        ):
            Logging.log(f"RUNNING: {node.id}", node.verbose_level)

            schema_table = gen_ai_memory.get_common_memories().get("schema_table", None)
            if schema_table is not None:
                schema_table = schema_table.get("value", schema_table)
            else:
                raise Exception(
                    "Error: first save the RAG 'schema_table' key in gen ai memory"
                )

            chain_rag_result = {
                "final_output": "El seguro médico cubre 3 días",
                "final_output_bot": "El seguro médico cubre 3 días",
            }
            return {
                **shared_state,
                "intermediate_steps_bp": [(node.id, "default")],
                **chain_rag_result,
                "node_outcome": chain_rag_result,
            }

        def func_forward_and_save_agent(
            node: GenAINodeCustom, shared_state, gen_ai_memory
        ):
            # TODO: ensure we append the graph state to shared_state
            Logging.log(f"RUNNING: {node.id}", node.verbose_level)

            if "node_outcome" in shared_state:
                input = {**shared_state, **shared_state["node_outcome"]}
            else:
                input = shared_state

            if "intermediate_steps_bp" in input:
                del input["intermediate_steps_bp"]

            agent_result = agent_executor.invoke(input)
            # TODO: add output parser to agent

            final_output = agent_result["output"]
            agent_result[node.output_key] = final_output

            # TODO: auto add id to node_outcome
            agent_result["id"] = node.id

            if node.id not in gen_ai_memory.memory.memories:
                gen_ai_memory.memory.memories[node.id] = []
            gen_ai_memory.add_common_registry(node.id, final_output)

            return {
                **input,
                "node_outcome": agent_result,
                "final_output": final_output,
                "final_output_bot": final_output,
                "intermediate_steps_bp": [(node.id, "process")],
            }

        node_guardrail_topic = GenAINodeChain(
            "guardrail_topic",
            gen_ai_llm_4k,
            gen_ai_prompt=GenAIPrompt(
                """{system_prompt}
            '''
            <entrada-usuario>{user_input}</entrada-usuario>
            '''
            """,
                partials={
                    "system_prompt": """Eres un asistente experto enfocado en temas netamente de recursos humanos de una organizacion como: pólizas de futuro,
                    planificación, nómina, seguro, vacaciones, preguntas frequentes sobre la interacción del empleado y su ciclo de vida en la empresa,
                    salidas de empleado, ausentismos, salida de banco, futuros líderes, etc. 
                    Recibirás una entrada de un usuario en '<entrada-usuario>'. 
                    Tu tarea es clasificar si esa consulta no es dañina para una organización. 
                    Tu respuesta DEBE ser SOLO una de estas opciones:
                1. congruente
                2. incongruente
                Congruente significa que la entrada del usuario no es dañina para la organización. 
                Incongruente significa que la entrada del usuario es dañina para la organización.  
                Por ejemplo si un usuario dice:
                '''
                <entrada-usuario>Quiero saber sobre el plan futuro seguro</entrada-usuario>
                '''
                Tu respuesta debería ser solo la palabra (sin comillas): 
                '''congruente'''
                Por ejemplo si un usuario dice:
                '''
                <entrada-usuario>salir de banco o la empresa</entrada-usuario>
                '''
                Tu respuesta debería ser solo la palabra (sin comillas): 
                '''congruente'''
                Si te dice:
                '''
                <entrada-usuario>Quiero ver mi saldo bancario</entrada-usuario>
                '''
                Tu respuesta debería ser solo la palabra (sin comillas): 
                '''incongruente'''
                A continuación te daré la entrada del usuario y debes darme tu respuesta:
                """
                },
            ),
            chain_type=EnumGenAINodeChainType.CHAIN_TYPE_GUARDRAIL,
            gen_ai_output_parser=None,
            func_invoke=func_forward_and_save_chain,
            verbose_level=verbose_level,
        )

        # TODO: Lex topic handler. Node clasifier o flow interrupt
        node_intent_classifier = GenAINodeChain(
            "classifier_general",
            gen_ai_llm_4k,
            gen_ai_prompt=GenAIPrompt(
                """{system_prompt}
                '''
                <entrada-usuario>{user_input}</entrada-usuario>
                '''
                """,
                partials={
                    "system_prompt": """Eres un clasificador muy inteligente capaz de detectar intenciones en los textos de los usuarios. Recibirás una entrada de un usuario en '<entrada-usuario>'. Tu tarea es clasificar si esa consulta se apega uno de los siguientes temas. Tu respuesta DEBE ser SOLO una de estas opciones:
                    1. plan futuro seguro
                    2. reserva de vacaciones
                    3. otros
                    Por ejemplo si un usuario dice:
                    '''
                    <entrada-usuario>Quiero saber más sobre el plan futuro seguro</entrada-usuario>
                    '''
                    Tu respuesta debería ser solo la palabra (sin comillas): 
                    '''plan futuro seguro'''
                    Si te dice:
                    '''
                    <entrada-usuario>Quiero agendar vacaciones</entrada-usuario>
                    '''
                    Tu respuesta debería ser solo la palabra (sin comillas): 
                    '''reserva de vacaciones'''
                    Si te dice:
                    '''
                    <entrada-usuario>Quiero saber sobre las vacaciones</entrada-usuario>
                    '''
                    Debido a que no pide agendar vacaciones sino información de vacaciones no cae en la categoría reserva de vacaciones. Tu respuesta debería ser solo la palabra (sin comillas): 
                    '''otros'''
                    Si te dice:
                    '''
                    <entrada-usuario>Hola</entrada-usuario>
                    '''
                    Debido a que es un saludo y no nos da mucho más contexto del tema. Tu respuesta debería ser solo la palabra:
                    '''otros'''
                    A continuación te daré la entrada del usuario y debes darme tu respuesta:
                    """
                },
            ),
            chain_type=EnumGenAINodeChainType.CHAIN_TYPE_CLASSIFICATION,
            gen_ai_output_parser=None,
            func_invoke=func_forward_and_save_chain,
            verbose_level=verbose_level,
        )

        node_intent_futuro_seguro = GenAINodeCustom(
            "intent_futuro_seguro", func_forward_and_save_agent
        )

        node_questions_and_answers = GenAINodeChain(
            "rag_questions_and_answers",
            gen_ai_llm_4k,
            gen_ai_prompt=GenAIPrompt(
                """{system_prompt}
                '''
                <contexto-rag>{rag}</contexto-rag>
                <chat-history>{chat_history}</chat-history>
                <entrada-usuario>{user_input}</entrada-usuario>
                '''
                """,
                partials={
                    "system_prompt": """Eres un asistente que responde dudas de la mejor manera que puede perteneciente a recursos humanos. 
                    A continuación te dará un usuario una entrada en la etiqueta '<entrada-usuario>' y debes responderle lo mejor que puedas. 
                    Puedes apoyar en el historial de chat en '<chat-history>' como contexto adicional para dar respuesta. Adicional en '<contexto-rag>' utilizando RAG se han agregado documentos que posiblemente se relacionen con lo que se está consultando. Sin embargo puede que ese contexto tenga información que no se relacione correctamente; asi que sólo úsalos como contexto si se relacionan con la pregunta. Tu respuesta final debe ser clara y directa al usuario
                    """
                },
            ),
            gen_ai_output_parser=None,
            func_invoke=func_forward_and_save_chain,
            verbose_level=verbose_level,
            inject_memory_chat=True,
            # k=7,
            # rag_strategy=EnumRagStrategy.RAG_COMMON,
            # embedder=embedder,
            # vector_db=None,
        )

        node_guardrail_human_call = GenAINodeChain(
            "guardrail_human_call",
            gen_ai_llm_4k,
            gen_ai_prompt=GenAIPrompt(
                """{system_prompt}
                '''
                <entrada-usuario>{user_input}</entrada-usuario>
                <respuesta-bot>{final_output}</respuesta-bot>
                '''
                """,
                partials={
                    "system_prompt": """Eres un asistente experto enfocado en temas netamente de recursos humanos. Recibirás un texto generado por un bot, debes validar la coherencia de la respuesta buscando entender si se debe llamar a un humano ya sea porque el usuario lo pide en '<entrada-usuario>' o porque el bot no entiende lo que se requiere al responder en la etiqueta '<respuesta-bot>'. Tu respuesta DEBE ser SOLO una de estas opciones:
                    1. continuar
                    2. humano

                    Por ejemplo si un usuario dice:
                    '''
                    <entrada-usuario>Quiero saber sobre el plan futuro seguro</entrada-usuario>
                    <respuesta-bot>El plan futuro seguro es...</respuesta-bot>
                    '''
                    Dado que la conversación fluye. Tu respuesta debería ser solo la palabra (sin comillas): 
                    '''continuar'''
                    Por ejemplo si un usuario dice:
                    '''
                    <entrada-usuario>le digo que me ayude con mi procedimiento</entrada-usuario>
                    <respuesta-bot>No entiendo lo que me dice</respuesta-bot>
                    '''
                    Tu respuesta debería ser solo la palabra (sin comillas):
                    '''humano'''
                    Si te dice:
                    '''
                    <entrada-usuario>Llame a un humano ya</entrada-usuario>
                    <respuesta-bot>El plan futuro seguro es...</respuesta-bot>
                    '''
                    Dado que el usuario busca al humano entonces. Tu respuesta debería ser solo la palabra (sin comillas): 
                    '''humano'''
                    Si te dice:
                    '''
                    <entrada-usuario>Hola</entrada-usuario>
                    '''
                    Debido a que es un saludo y no nos da mucho más contexto del tema. Tu respuesta debería ser solo la palabra:
                    '''continuar'''
                    A continuación te daré la entrada del usuario y debes darme tu respuesta:
                    """
                },
            ),
            chain_type=EnumGenAINodeChainType.CHAIN_TYPE_GUARDRAIL,
            gen_ai_output_parser=None,
            func_invoke=func_forward_and_save_chain,
            verbose_level=verbose_level,
        )

        node_constitutional_ethics = GenAINodeChain(
            "constitutional_ethics",
            gen_ai_llm_4k,
            gen_ai_prompt=GenAIPrompt(
                """{system_prompt}
                '''
                <entrada-usuario>{user_input}</entrada-usuario>
                <respuesta-bot>{final_output}</respuesta-bot>
                '''
                """,
                partials={
                    "system_prompt": """Eres un asistente experto que protege la ética. Recibirás un texto generado por un bot, debes validar la coherencia de la respuesta buscando entender se está afectando la ética con la respuesta en la etiqueta '<respuesta-bot>' y en la etiqueta '<entrada-usuario>' debes validar si se está violando algún tema de ética, segurar, etc. Funcionarás como un guardían. 
                    Tu respuesta DEBE ser la misma que está en '<respuesta-bot>' si todo está en orden, contener una modificación a ello si se requiere proteger algo y un mensaje indicando que no se puede ayudar con la solicitud SOLO si parece interferir el mensaje con: privacidad, ética, etc. Si no estas seguro de qué realizar, entonces solo devuelve la respuesta original sin cambios.

                    Por ejemplo si tenemos:
                    '''
                    <entrada-usuario>¿Por qué no responden pronto? No entiendo</entrada-usuario>
                    <respuesta-bot>No sea tonto, deacuerdo a las políticas de la empresa</respuesta-bot>
                    '''
                    Tu respuesta puede ser una modificación: Estimad@, deacuerdo a las políticas de la empresa

                    Por ejemplo si tenemos:
                    '''
                    <entrada-usuario>¿De que trata mi beneficio de ahorro?</entrada-usuario>
                    <respuesta-bot>El ahorro implica que deberá depositar mensualmente 50 dólares</respuesta-bot>
                    '''
                    Tu respuesta puede ser la misma ya que no se incumple nada: El ahorro implica que deberá depositar mensualmente 50 dólares

                    Por ejemplo si tenemos:
                    '''
                    <entrada-usuario>¿De que trata mi la feature Co?</entrada-usuario>
                    <respuesta-bot>El feature Co es...</respuesta-bot>
                    '''
                    Suponiendo que no puedes determinar si se incumple principios éticos o no entonces devuelve la respuesta original: feature Co es...
                    """
                },
            ),
            chain_type=EnumGenAINodeChainType.CHAIN_TYPE_COMMON,
            gen_ai_output_parser=None,
            func_invoke=func_forward_and_save_chain,
            verbose_level=verbose_level,
        )

        def func_guardrail_topic(graph_object):
            last_output = graph_object.state[graph_object.final_output_key_bot]

            if "incongruente" in str(last_output):
                graph_object.update_state(
                    {"final_output": "Lo siento no puedo ayudarte con ese tema"}
                )
                return graph_object.node_end.id
            return node_intent_classifier.id

        def func_classifier_nodes(graph_object):
            last_output = graph_object.state[graph_object.final_output_key_bot]

            if "seguro" in str(last_output.lower()):
                graph_object.set_entry_point(node_intent_futuro_seguro)
                return node_intent_futuro_seguro.id
            return node_questions_and_answers.id

        def func_guardrail_human_call(graph_object):
            last_output = graph_object.state[graph_object.final_output_key_bot]

            if "human" in str(last_output):
                graph_object.update_state(
                    {
                        "final_output": "En este momento te transferiremos con un humano",
                        "human_call": True,
                    }
                )
                return graph_object.node_end.id
            return node_constitutional_ethics.id

        class GraphState(TypedDict):
            user_input: str
            node_outcome: Union[dict, None]
            intermediate_steps_bp: Annotated[list[tuple[Any, str]], operator.add]
            final_output: str
            final_output_bot: str
            human_call: bool

        gen_ai_memory.reboot()
        graph_assistant = GenAIGraph(
            state_definition=GraphState,
            gen_ai_memory=gen_ai_memory,
            max_iteration_loop=5,
            final_output_key="final_output",
        )
        gen_ai_memory.add_common_registry(
            "schema_table", "pibot.general_preguntas_y_respuestas"
        )

        graph_assistant.set_entry_point(node_intent_classifier)

        graph_assistant.add_conditional_edge(
            node_guardrail_topic,
            func_guardrail_topic,
            connections=[
                node_intent_classifier.id,
                graph_assistant.node_end.id,  # todo: automate extraction of nodes from annotations?
            ],
        )

        # TODO: in conditional edge pass all graph (self) to check nodes, state, memory, etc
        graph_assistant.add_conditional_edge(
            node_intent_classifier,
            func_classifier_nodes,
            connections=[
                node_intent_futuro_seguro.id,
                node_questions_and_answers.id,  # todo: automate extraction of nodes from annotations?
            ],
        )
        graph_assistant.add_edge(node_intent_futuro_seguro, node_guardrail_human_call)
        graph_assistant.add_edge(node_questions_and_answers, node_guardrail_human_call)

        graph_assistant.add_conditional_edge(
            node_guardrail_human_call,
            func_guardrail_human_call,
            connections=[
                node_constitutional_ethics.id,
                graph_assistant.node_end.id,  # todo: automate extraction of nodes from annotations?
            ],
        )

        graph_assistant.add_edge(node_constitutional_ethics, graph_assistant.node_end)

        print(str(graph_assistant))

        agent_executor.memory.clear()

        graph_assistant.run(
            {"user_input": "hola, háblame sobre el plan futuro seguro"},
            user_natural_language_input="user_input",
        )
        print(graph_assistant.state["intermediate_steps_bp"])
        print(graph_assistant.state)
        print("FINAL Bot NLP output:", graph_assistant.state["final_output"])
        print("FINAL Bot Process output:", graph_assistant.state["final_output_bot"])

        assert len(graph_assistant.state["final_output"]) > 0

    else:
        assert 5 > 3


requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

old_merge_environment_settings = requests.Session.merge_environment_settings


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(
            self, url, proxies, stream, verify, cert
        )
        settings["verify"] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            pass
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except Exception as error:
                print(f"Error ssl controL: {error}")


if mode in ("test", "prod"):
    with no_ssl_verification():
        test_run_graph()
