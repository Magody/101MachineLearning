from ai_framework.GenAILLM import GenAILLM
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.GenAIPrompt import EnumPromptsBase, GenAIPrompt
from ai_framework.prefabs.PrefabNode import PrefabNode
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging


class PrefabNodeTranscoder(PrefabNode):
    """
    The PrefabNodeTranscoder class is a subclass of the PrefabNode class, which is part of the ai_framework module.
    This class is responsible for transcoding code from one programming language to another.

    Args:
        gen_ai_llm (GenAILLM): An instance of the GenAILLM class, which is used for language modeling.
        programming_language_origin (str): The programming language of the input code.
        programming_language_dest (str): The programming language to which the code should be transcoded.
        project_or_class_summary (str, optional): A summary of the project or class, which can be used to provide additional context for the transcoding process.
        node_id_next (str, optional): The ID of the next node in the node chain.
        input_maps (dict, optional): A dictionary that maps input keys to their corresponding values.
        code_key (str, optional): The key used to access the code in the input dictionary.
        verbose_level (EnumLogs, optional): The level of verbosity for logging.

    Attributes:
        project_or_class_summary (str): The summary of the project or class.
        node (GenAINodeChain): An instance of the GenAINodeChain class, which is used to manage the code transcoding process.
    """

    def __init__(
        self,
        gen_ai_llm: GenAILLM,
        programming_language_origin: str,
        programming_language_dest: str,
        project_or_class_summary: str = "",
        node_id_next: str = None,
        input_maps: dict = {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        },
        code_key: str = "code",
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
    ) -> None:
        name = "node_transcoder"
        super().__init__(name, gen_ai_llm, None, input_maps, verbose_level, node_next=node_id_next)

        self.project_or_class_summary = project_or_class_summary

        self.node = GenAINodeChain(
            name,
            gen_ai_llm,
            gen_ai_prompt=GenAIPrompt(
                EnumPromptsBase.PROMPT_TEMPLATE_CODE_TRANSCODER.value.replace("{code}", "{" + code_key + "}"),
                partials={
                    "programming_language_origin": programming_language_origin,
                    "programming_language_dest": programming_language_dest,
                }
            ),
            gen_ai_output_parser=None,
            func_invoke=self.func_invoke,
        )

    def func_invoke(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
        """
        The func_invoke method is responsible for running the code transcoding process.

        Args:
            node (GenAINodeChain): The current node in the node chain.
            shared_state (dict): A dictionary containing the shared state of the node chain.
            gen_ai_memory (GenAIMemory): An instance of the GenAIMemory class, which is used to manage the memory of the AI system.

        Returns:
            dict: A dictionary containing the updated shared state, the result of the code transcoding process, and the intermediate steps of the process.
        """
        # TODO: remove need of 'node' in parameter since in new architecture self is enough
        Logging.log(f"RUNNING: {self.node.id}", self.node.verbose_level)

        there_exist_summary = len(self.project_or_class_summary.strip()) > 1
        extra_partials: dict = {}
        if there_exist_summary:
            extra_partials["extra_prompt"] = f"""Adicionalmente se agrega contexto de la clase en general que puede o no ser de ayuda en la etiqueta '<resumen-clase>' para que lo uses de apoyo.
            <resumen-clase>{self.project_or_class_summary}</resumen-clase>"""

        else:
            # the variable MUST be in shared_state
            extra_partials["extra_prompt"] = f"""Adicionalmente se agrega contexto de la clase en general que puede o no ser de ayuda en la etiqueta '<resumen-clase>' para que lo uses de apoyo.
            <resumen-clase>{shared_state['project_or_class_summary']}</resumen-clase>"""

        full_state = {**shared_state, **extra_partials}

        chain_result = self.node.get_chain_result(full_state, gen_ai_memory)

        return {
            **shared_state,
            **chain_result,
            "intermediate_steps_bp": [(self.node.id, "default")],
            # TODO: Bug in GenAINodeChain, who extract data from node_outcome and edge changing state not reflect due to override
            "node_outcome": chain_result,
        }
