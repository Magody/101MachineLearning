from ai_framework.GenAILLM import GenAILLM
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.GenAINodeChain import GenAINodeChain
from ai_framework.GenAIPrompt import EnumPromptsBase, GenAIPrompt
from ai_framework.prefabs.PrefabNode import PrefabNode
from ai_framework.LoggingAndTelemetry import EnumLogs, Logging


class PrefabNodeCodeUnitTester(PrefabNode):
    """
    A class that represents a prefab node for code unit testing.

    This class is responsible for generating code unit tests for a given programming language.

    Attributes:
        gen_ai_llm (GenAILLM): An instance of the GenAILLM class, which is used for language model generation.
        programming_language_for_unit_testing (str): The programming language for which the unit tests will be generated.
        project_or_class_summary (str): A summary of the project or class being tested.
        node_id_next (str): The ID of the next node in the chain.
        input_maps (dict): A dictionary mapping input keys to their corresponding values.
        code_key (str): The key for the code in the input maps.
        verbose_level (EnumLogs): The verbosity level for logging.
    """

    def __init__(
        self,
        gen_ai_llm: GenAILLM,
        programming_language_for_unit_testing: str,
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
        """
        Initializes the PrefabNodeCodeUnitTester class.

        Args:
            gen_ai_llm (GenAILLM): An instance of the GenAILLM class.
            programming_language_for_unit_testing (str): The programming language for which the unit tests will be generated.
            project_or_class_summary (str): A summary of the project or class being tested.
            node_id_next (str): The ID of the next node in the chain.
            input_maps (dict): A dictionary mapping input keys to their corresponding values.
            code_key (str): The key for the code in the input maps.
            verbose_level (EnumLogs): The verbosity level for logging.
        """
        name = "node_code_unit_testing_generator"
        super().__init__(name, gen_ai_llm, None, input_maps, verbose_level, node_next=node_id_next)

        self.project_or_class_summary = project_or_class_summary

        self.node = GenAINodeChain(
            name,
            gen_ai_llm,
            gen_ai_prompt=GenAIPrompt(
                EnumPromptsBase.PROMPT_TEMPLATE_CODE_UNIT_TESTER.value.replace("{code}", "{" + code_key + "}"),
                partials={
                    "programming_language_for_unit_testing": programming_language_for_unit_testing,
                }
            ),
            gen_ai_output_parser=None,
            func_invoke=self.func_invoke,
        )

    def func_invoke(self, node, shared_state: dict, gen_ai_memory: GenAIMemory):
        """
        Invokes the code unit testing functionality.

        Args:
            node (GenAINodeChain): The current node in the chain.
            shared_state (dict): The shared state dictionary.
            gen_ai_memory (GenAIMemory): An instance of the GenAIMemory class.

        Returns:
            dict: The updated shared state dictionary.
        """
        # TODO: remove need of 'node' in parameter since in new architecture self is enough
        Logging.log(f"RUNNING: {self.node.id}", self.node.verbose_level)

        there_exist_summary = len(self.project_or_class_summary.strip()) > 1
        extra_partials: dict = {}
        if there_exist_summary:
            extra_partials["extra_prompt"] = f"""Puede que requieras un contexto adicional del proyecto o clase. Ese contexto estará en la etiqueta <resumen-clase>.
            <resumen-clase>{self.project_or_class_summary}</resumen-clase>"""

        else:
            # the variable MUST be in shared_state
            extra_partials["extra_prompt"] = f"""Puede que requieras un contexto adicional del proyecto o clase. Ese contexto estará en la etiqueta <resumen-clase>.
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
