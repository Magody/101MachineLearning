from ai_framework.GenAILLM import GenAILLM
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.GenAIPrompt import GenAIPrompt, EnumPromptsBase
from ai_framework.prefabs.PrefabNode import PrefabNode
from ai_framework.LoggingAndTelemetry import EnumLogs


class PrefabNodeCodeDocumenter(PrefabNode):
    """
    A prefab node that generates documentation for a given code snippet in a specified programming language.

    Args:
        gen_ai_llm (GenAILLM): The language model to be used for generating the documentation.
        programming_language_to_doc (str): The programming language of the code to be documented.
        node_id_next (str, optional): The ID of the next node in the chain. Defaults to None.
        input_maps (dict, optional): A dictionary mapping input keys to their corresponding values. Defaults to {'user_input': 'user_input', 'final_output': 'final_output', 'final_output_bot': 'final_output_bot'}.
        code_key (str, optional): The key for the code input. Defaults to 'code'.
        verbose_level (EnumLogs, optional): The verbosity level for logging. Defaults to EnumLogs.LOG_LEVEL_INFO.

    Attributes:
        name (str): The name of the node.
        node (GenAINodeChain): The underlying GenAINodeChain object that handles the documentation generation.
    """

    def __init__(
        self,
        gen_ai_llm: GenAILLM,
        programming_language_to_doc: str,
        node_id_next: str = None,
        input_maps: dict = {
            "user_input": "user_input",
            "final_output": "final_output",
            "final_output_bot": "final_output_bot",
        },
        code_key: str = "code",
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
    ) -> None:
        name = "node_code_documenter"
        super().__init__(name, gen_ai_llm, None, input_maps, verbose_level, node_next=node_id_next)

        self.node = GenAINodeChain(
            name,
            gen_ai_llm,
            gen_ai_prompt=GenAIPrompt(
                EnumPromptsBase.PROMPT_TEMPLATE_CODE_DOCUMENTER.value.replace("{code}", "{" + code_key + "}"),
                partials={
                    "programming_language_to_doc": programming_language_to_doc
                }
            ),
            gen_ai_output_parser=None,
            func_invoke=self.func_invoke
        )
