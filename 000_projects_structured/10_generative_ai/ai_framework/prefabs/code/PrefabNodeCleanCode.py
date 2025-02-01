from ai_framework.GenAINode import GenAINodeCustom
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.prefabs.PrefabNode import PrefabNode
from ai_framework.LoggingAndTelemetry import EnumLogs
import re


class PrefabNodeCleanCode(PrefabNode):
    """
    A PrefabNode class that cleans the code input by removing excessive whitespace and newlines.

    Attributes:
        node_id_next (str): The ID of the next node in the AI framework.
        input_maps (dict): A dictionary mapping input keys to their corresponding values.
        verbose_level (EnumLogs): The logging level for the node.
    """

    def __init__(
        self,
        node_id_next: str = None,
        input_maps: dict = {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        },
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
    ) -> None:
        """
        Initializes the PrefabNodeCleanCode object.

        Args:
            node_id_next (str): The ID of the next node in the AI framework.
            input_maps (dict): A dictionary mapping input keys to their corresponding values.
            verbose_level (EnumLogs): The logging level for the node.
        """
        name = "node_clean_code"
        super().__init__(name, None, None, input_maps, verbose_level, node_next=node_id_next)

        self.node = GenAINodeCustom(name, self.func_invoke)

    def func_invoke(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
        """
        Cleans the code input by removing excessive whitespace and newlines.

        Args:
            node (GenAINodeCustom): The current node in the AI framework.
            shared_state (dict): The shared state dictionary.
            gen_ai_memory (GenAIMemory): The AI memory object.

        Returns:
            dict: A dictionary containing the cleaned code.
        """
        result = re.sub(r'\s+', ' ', shared_state["code"].replace("\n", "#NEWLINE#")).replace("#NEWLINE#", "\n")
        result = re.sub('\\n+', '\n', result)
        # result = re.sub('\\n{3,}', '\n\n', result)
        return {"code": result.strip()}
