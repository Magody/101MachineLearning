import operator
from typing import Annotated, Any, Union
from typing_extensions import TypedDict
from ai_framework.GenAIGraph import GenAIGraph
from ai_framework.GenAILLM import GenAILLM
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.prefabs.FactoryGraph import FactoryGraph

from ai_framework.prefabs.code.PrefabNodeStructureAnalizer import PrefabNodeCodeStructureAnalizer
from ai_framework.prefabs.code.PrefabNodeCleanCode import PrefabNodeCleanCode
from ai_framework.prefabs.code.PrefabNodeCodeDocumenter import PrefabNodeCodeDocumenter
from ai_framework.prefabs.code.PrefabNodeCodeUnitTester import PrefabNodeCodeUnitTester
from ai_framework.prefabs.code.PrefabNodeTranscoder import PrefabNodeTranscoder
from enum import Enum


class EnumFactoryGraphTranscoderMode(Enum):
    """
    Enum to define the different modes of the FactoryGraphTranscoder.

    Attributes:
        GRAPH_FULL (int): Includes all the nodes in the graph.
        GRAPH_ONLY_TRANSCODER (int): Includes only the transcoder node in the graph.
        GRAPH_ONLY_STRUCTURE_ANALYZER (int): Includes only the structure analyzer node in the graph.
    """
    GRAPH_FULL = 0
    GRAPH_ONLY_TRANSCODER = 1
    GRAPH_ONLY_STRUCTURE_ANALYZER = 2


class FactoryGraphTranscoder(FactoryGraph):
    """
    A factory class that creates a GenAIGraph for code transcoding and documentation.

    This class defines the nodes and the connections between them based on the selected mode.

    Args:
        gen_ai_memory (GenAIMemory): The memory object to be used by the graph.
        programming_language_origin (str): The programming language of the input code.
        programming_language_dest (str): The programming language to which the code should be transcoded.
        gen_ai_llm_16k_plus (GenAILLM): The large language model to be used for transcoding and documentation.
        gen_ai_llm_128k_plus (GenAILLM): The larger language model to be used for code structure analysis.
        enum_include_structure (EnumFactoryGraphTranscoderMode, optional): The mode to be used for the graph. Defaults to EnumFactoryGraphTranscoderMode.GRAPH_FULL.

    Returns:
        GenAIGraph: The created graph.
    """

    @staticmethod
    def create(
        gen_ai_memory: GenAIMemory,
        programming_language_origin: str,
        programming_language_dest: str,
        gen_ai_llm_16k_plus: GenAILLM,
        gen_ai_llm_128k_plus: GenAILLM,
        enum_include_structure: EnumFactoryGraphTranscoderMode = EnumFactoryGraphTranscoderMode.GRAPH_FULL
    ):
        # Nodes definition
        prefab_node_clean = PrefabNodeCleanCode()

        prefab_node_structure_analizer = PrefabNodeCodeStructureAnalizer(
            gen_ai_llm_128k_plus,
            programming_language_origin=programming_language_origin,
            code_key="code"
        )

        code_key_transcoder = "code"
        if enum_include_structure.value == EnumFactoryGraphTranscoderMode.GRAPH_FULL.value:
            code_key_transcoder = "final_output_bot"

        prefab_node_transcoder = PrefabNodeTranscoder(
            gen_ai_llm_16k_plus,
            programming_language_origin=programming_language_origin,
            programming_language_dest=programming_language_dest,
            code_key=code_key_transcoder
        )

        prefab_node_code_documenter = PrefabNodeCodeDocumenter(
            gen_ai_llm_16k_plus,
            programming_language_to_doc=programming_language_dest,
            code_key="final_output_bot"
        )

        prefab_node_code_unit_tester = PrefabNodeCodeUnitTester(
            gen_ai_llm_16k_plus,
            programming_language_for_unit_testing=programming_language_dest,
            code_key="final_output_bot"
        )

        node_clean = prefab_node_clean.node
        node_structure_analizer = prefab_node_structure_analizer.node
        node_transcoder = prefab_node_transcoder.node
        node_code_documenter = prefab_node_code_documenter.node
        node_code_unit_tester = prefab_node_code_unit_tester.node

        # Graph definition
        class GraphState(TypedDict):
            code: str
            code_unmodified: str
            node_outcome: Union[dict, None]
            intermediate_steps: Annotated[list[tuple[Any, str]], operator.add]
            final_output: str
            project_or_class_summary: str  # It could auto-generated in running time

        gen_ai_memory.reboot()

        graph = GenAIGraph(
            state_definition=GraphState,
            gen_ai_memory=gen_ai_memory,
            max_iteration_loop=5,
            final_output_key="final_output"
        )

        if enum_include_structure.value == EnumFactoryGraphTranscoderMode.GRAPH_FULL.value:
            graph.add_edge(node_clean, node_structure_analizer)
            graph.add_edge(node_structure_analizer, node_transcoder)
            graph.add_edge(node_transcoder, node_code_documenter)
            graph.add_edge(node_code_documenter, node_code_unit_tester)
            graph.add_edge(node_code_unit_tester, graph.node_end)

        elif enum_include_structure.value == EnumFactoryGraphTranscoderMode.GRAPH_ONLY_TRANSCODER.value:
            graph.add_edge(node_clean, node_transcoder)
            graph.add_edge(node_transcoder, node_code_documenter)
            graph.add_edge(node_code_documenter, node_code_unit_tester)
            graph.add_edge(node_code_unit_tester, graph.node_end)

        elif enum_include_structure.value == EnumFactoryGraphTranscoderMode.GRAPH_ONLY_STRUCTURE_ANALYZER.value:
            graph.add_edge(node_clean, node_structure_analizer)
            graph.add_edge(node_structure_analizer, graph.node_end)

        graph.set_entry_point(node_clean)
        return graph
