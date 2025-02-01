from ai_framework.GenAILLM import GenAILLM
from ai_framework.GenAINode import GenAINodeCustom
from ai_framework.GenAINodeChain import EnumGenAINodeChainType, GenAINodeChain
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAIGraph import GenAIGraph
from ai_framework.GenAIPrompt import GenAIPrompt
from ai_framework.prefabs.PrefabNode import PrefabNode
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging


class PrefabNodeFeedbackSurvey(PrefabNode):
    """
    A prefab node that handles the feedback survey functionality.

    Args:
        gen_ai_llm (GenAILLM): The GenAILLM instance to be used.
        input_maps (dict, optional): A dictionary mapping input keys to shared state keys. Defaults to {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        }.
        verbose_level (EnumLogs, optional): The verbose level for logging. Defaults to EnumLogs.LOG_LEVEL_INFO.

    Attributes:
        node (GenAINodeCustom): The custom GenAINode instance for this prefab node.
    """

    def __init__(
        self,
        gen_ai_llm: GenAILLM,
        input_maps: dict = {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        },
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
    ) -> None:
        id = "node_feedback"
        super().__init__(id, gen_ai_llm, dict(), input_maps, verbose_level)

        self.node = GenAINodeCustom(id, self.func_invoke)

    def func_invoke(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
        """
        The function that is invoked when the node is executed.

        Args:
            node (GenAINode): The node instance.
            shared_state (dict): The shared state dictionary.
            gen_ai_memory (GenAIMemory): The GenAIMemory instance.

        Returns:
            dict: The updated shared state dictionary.
        """
        Logging.log(f"RUNNING: {self.node.id}", self.node.verbose_level)

        final_output = shared_state[self.input_maps["final_output"]]
        # In order to use this, previously a node MUST generate a NLP response in 'final_response' key
        chain_result = {
            self.input_maps[
                "final_output"
            ]: f"""{final_output}.\nTu experiencia es importante para nosotros, por favor si tienes alguna sugerencia sobre mi servicio coméntamela""",
            "esperando_feedback": True,
        }

        return {
            **shared_state,
            "intermediate_steps_bp": [(self.node.id, "Augmentating feedback")],
            **chain_result,
            "node_outcome": chain_result,
        }


class PrefabNodeFeedbackChecker(PrefabNode):
    """
    A prefab node that checks the user's feedback.

    Args:
        gen_ai_llm (GenAILLM): The GenAILLM instance to be used.
        node_id_yes (str): The ID of the node to be executed if the feedback is positive.
        node_id_no (str): The ID of the node to be executed if the feedback is negative.
        node_id_uncertain (str): The ID of the node to be executed if the feedback is uncertain.
        input_maps (dict, optional): A dictionary mapping input keys to shared state keys. Defaults to {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        }.
        verbose_level (EnumLogs, optional): The verbose level for logging. Defaults to EnumLogs.LOG_LEVEL_INFO.

    Attributes:
        node (GenAINodeChain): The GenAINodeChain instance for this prefab node.
    """

    def __init__(
        self,
        gen_ai_llm: GenAILLM,
        node_id_yes: str,
        node_id_no: str,
        node_id_uncertain: str,
        input_maps: dict = {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        },
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
    ) -> None:
        name = "node_classifier_feedback_checker"
        nodes = {
            "node_id_yes": node_id_yes,
            "node_id_no": node_id_no,
            "node_id_uncertain": node_id_uncertain,
        }
        super().__init__(name, gen_ai_llm, nodes, input_maps, verbose_level)

        self.node = GenAINodeChain(
            name,
            gen_ai_llm,
            gen_ai_prompt=GenAIPrompt(
                """{system_prompt}

                Ahora la entrada del usuario es la siguiente, clasifícala:
                '''
                <entrada-usuario>{user_input}</entrada-usuario>
                '''
                """,
                partials={
                    "system_prompt": """Recibirás una entrada de un usuario en la etiqueta '<entrada-usuario>'.
                    Tu tarea es determinar si se trata de un feedback o no, por lo que tus respuestas solo pueden ser UNA SOLA PALABRA de las que te listo a continuación:
                    1. Si
                    2. No
                    3. Incierto

                    A continuación te daré algunos ejemplos:

                    '''
                    <entrada-usuario>El número de mi identificación es</entrada-usuario>
                    '''
                    Dado que habla de su identificación entonces tu respuesta sería: 'No'

                    '''
                    <entrada-usuario>Me parece un mal servicio</entrada-usuario>
                    '''
                    Dado que habla de una atención de servicio entonces tu respuesta sería: 'Si'

                    '''
                    <entrada-usuario>Si, eso podría funcionar</entrada-usuario>
                    '''
                    Dado que no está claro si se habla del servicio o no entonces tu respuesta sería: 'Incierto'

                    '''
                    <entrada-usuario>Está bien, gracias</entrada-usuario>
                    '''
                    Dado que da un agradecimiento, tu respuesta sería: 'Incierto'

                    Es importante que no vuelvas a incluir la entrada del usuario, sino que solo responde directamente en una sola palabra.
                    """
                },
            ),
            chain_type=EnumGenAINodeChainType.CHAIN_TYPE_CLASSIFICATION,
            gen_ai_output_parser=None,
            func_invoke=self.func_invoke,
            verbose_level=self.verbose_level,
        )

    def func_edge_connection(self, graph_object: GenAIGraph):
        """
        The function that determines the next node to be executed based on the feedback classification.

        Args:
            graph_object (GenAIGraph): The GenAIGraph instance.

        Returns:
            str: The ID of the next node to be executed.
        """
        user_input = graph_object.state[self.input_maps['user_input']]
        last_output = (
            str(graph_object.state[graph_object.final_output_key_bot])
            .lower()
            .strip()
            .replace("í", "i")
        )
        Logging.log(
            f"EDGE function, last output: {last_output}",
            self.node.verbose_level,
            minimum_verbose_level=EnumLogs.LOG_LEVEL_DEBUG,
        )

        if "si" in last_output:
            graph_object.update_state({
                "esperando_feedback": False,
                "feedback": user_input,
                # TODO: Bug in GenAINodeChain, who extract data from node_outcome and edge changing state not reflect due to override
                "node_outcome": {
                    "esperando_feedback": False,
                    "feedback": user_input,
                }
            })
            return self.nodes["node_id_yes"]

        if "no" in last_output:
            return self.nodes["node_id_no"]

        return self.nodes["node_id_uncertain"]
