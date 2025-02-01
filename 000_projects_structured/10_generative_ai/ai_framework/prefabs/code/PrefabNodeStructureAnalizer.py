from ai_framework.GenAILLM import GenAILLM
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.GenAIPrompt import EnumPromptsBase, GenAIPrompt
from ai_framework.prefabs.PrefabNode import PrefabNode
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging


class PrefabNodeCodeStructureAnalizer(PrefabNode):
    """This class is responsible for analyzing the structure of a code snippet and providing a summary of the project or class.

    Attributes:
        gen_ai_llm (GenAILLM): An instance of the GenAILLM class, which is used for language model-based tasks.
        programming_language_origin (str): The programming language of the code snippet.
        node_id_next (str, optional): The ID of the next node in the node chain.
        input_maps (dict, optional): A dictionary mapping input keys to their corresponding values.
        code_key (str, optional): The key for the code snippet in the input maps.
        verbose_level (EnumLogs, optional): The verbosity level for logging.

    Methods:
        func_invoke(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
            Runs the code structure analysis and returns the result.
    """

    def __init__(
        self,
        gen_ai_llm: GenAILLM,
        programming_language_origin: str,
        node_id_next: str = None,
        input_maps: dict = {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        },
        code_key: str = "code",
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
    ) -> None:
        name = "node_structure_analizer"
        super().__init__(name, gen_ai_llm, None, input_maps, verbose_level, node_next=node_id_next)

        self.node = GenAINodeChain(
            name,
            gen_ai_llm,
            gen_ai_prompt=GenAIPrompt(
                EnumPromptsBase.PROMPT_TEMPLATE_CODE_STRUCTURE_ANALIZER.value.replace("{code}", "{" + code_key + "}"),
                partials={
                    "programming_language_origin": programming_language_origin
                }
            ),
            gen_ai_output_parser=None,
            func_invoke=self.func_invoke,
        )

    def func_invoke(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
        """
        Runs the code structure analysis and returns the result. It's overriding the abstract the father method

        Args:
            node (GenAINodeChain): The current node in the node chain.
            shared_state (dict): The shared state dictionary.
            gen_ai_memory (GenAIMemory): The GenAIMemory instance.

        Returns:
            dict: The updated shared state dictionary with the analysis result.
        """
        # TODO: remove need of 'node' in parameter since in new architecture self is enough
        Logging.log(f"RUNNING: {self.node.id}", self.node.verbose_level)

        chain_result = self.node.get_chain_result(shared_state, gen_ai_memory)

        final_output_bot = chain_result[self.input_maps["final_output_bot"]]

        chain_result["project_or_class_summary"] = final_output_bot

        return {
            **shared_state,
            **chain_result,
            "intermediate_steps_bp": [(self.node.id, "default")],
            # TODO: Bug in GenAINodeChain, who extract data from node_outcome and edge changing state not reflect due to override
            "node_outcome": chain_result,
        }
