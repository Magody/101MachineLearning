import sys
from ai_framework.prefabs.code.FactoryGraphTranscoder import (
    FactoryGraphTranscoder,
    EnumFactoryGraphTranscoderMode,
)
import os
from ai_framework.GenAILLM import (
    EnumGenAIModelsIdsBedrock,
    EnumGenAIPlatforms,
    GenAILLM,
)
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.LoggingAndTelemetry import EnumLogs
import pandas as pd

from ai_framework.test.utils import get_mode

mode = get_mode(sys.argv)


def test_graph_transcoder():
    if mode == "debug":
        pass
    elif mode in ("test", "prod"):

        verify_ssl = True
        verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_DEBUG

        path_workspace = os.getcwd()

        gen_ai_memory = GenAIMemory()

        platform_configuration_aws = {
            "aws_access_key_id": os.environ["aws_access_key_id"],
            "aws_secret_access_key": os.environ["aws_secret_access_key"],
            "region_name": os.environ["region_name"],
        }
        metadata = pd.read_csv(f"{path_workspace}/model_prices.csv", delimiter=";")

        gen_ai_llm_4k = GenAILLM(
            platform=EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK,
            model_id=EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_SONNET,
            parameters_inference={
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 0.1,
                "stop_sequences": ["User"],
                "top_k": 50,
            },
            platform_configuration=platform_configuration_aws,
            verify_ssl=verify_ssl,
            verbose_level=verbose_level,
            metadata=metadata,
        )

        gen_ai_llm_16k = GenAILLM(
            platform=EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK,
            model_id=EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_HAIKU,
            parameters_inference={
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 0.1,
                "stop_sequences": ["User"],
                "top_k": 50,
            },
            platform_configuration=platform_configuration_aws,
            verify_ssl=verify_ssl,
            verbose_level=verbose_level,
            metadata=metadata,
        )

        gen_ai_llm_128k = GenAILLM(
            platform=EnumGenAIPlatforms.PLATFORM_AWS_BEDROCK,
            model_id=EnumGenAIModelsIdsBedrock.MODEL_CLAUDE_V3_HAIKU,
            parameters_inference={
                "max_tokens": 4096,
                "temperature": 0,
                "top_p": 0.1,
                "stop_sequences": ["User"],
                "top_k": 50,
            },
            platform_configuration=platform_configuration_aws,
            verify_ssl=verify_ssl,
            verbose_level=verbose_level,
            metadata=metadata,
        )

        gen_ai_llm_4k.verbose_level = EnumLogs.LOG_LEVEL_NOTHING
        gen_ai_llm_4k.verbose = False
        gen_ai_llm_16k.verbose_level = EnumLogs.LOG_LEVEL_NOTHING
        gen_ai_llm_16k.verbose = False
        gen_ai_llm_128k.verbose_level = EnumLogs.LOG_LEVEL_NOTHING
        gen_ai_llm_128k.verbose = False

        sample_file = """import javalang

        from ai_framework.prefabs.code.parsers import BaseParser


        class ParserCodeJava(BaseParser):

            def __init__(self, code):
                super().__init__(code)

            def extract_tree(self):
                self.tree = javalang.parse.parse(self.code)
                self.package = self.tree.package.name
                self.imports = self.get_imports()
                self.classes = []

                for _, node_class in self.tree.filter(javalang.tree.ClassDeclaration):
                    start, end = self.__get_start_end_for_node(node_class)
                    class_definition = {
                        "type": "class",
                        "name": node_class.name,
                        "content": "",  # __get_string(java_code, start, end),
                        "fields": [],
                        "methods": []
                    }

                    for path_field, node_field in node_class.filter(javalang.tree.FieldDeclaration):
                        start, end = self.__get_start_end_for_node(node_field)
                        class_definition["fields"].append({
                            "type": "field",
                            "name": node_field.declarators[0].name,
                            "content": self.__get_string(start, end)
                        })

                    for _, node_method in self.tree.filter(javalang.tree.MethodDeclaration):
                        start, end = self.__get_start_end_for_node(node_method)
                        class_definition["methods"].append({
                            "type": "method",
                            "name": node_method.name,
                            "content": self.__get_string(start, end)
                        })

                    self.classes.append(class_definition)

            def get_imports(self):
                output = ""
                for import_path in self.tree.imports:
                    output += f"import {import_path.path};\n"
                return output

            def __get_start_end_for_node(self, node_to_find):
                start = None
                end = None
                for path, node in self.tree:
                    if start is not None and node_to_find not in path:
                        end = node.position
                        return start, end
                    if start is None and node == node_to_find:
                        start = node.position
                return start, end

            def __get_string(self, start, end):
                if start is None:
                    return ""

                # positions are all offset by 1. e.g. first line -> lines[0], start.line = 1
                end_pos = None

                if end is not None:
                    end_pos = end.line - 1

                lines = self.code.splitlines(True)
                string = "".join(lines[start.line:end_pos])
                string = lines[start.line - 1] + string

                # When the method is the last one, it will contain a additional brace
                if end is None:
                    left = string.count("{")
                    right = string.count("}")
                    if right - left == 1:
                        p = string.rfind("}")
                        string = string[:p]

                return string
        """

        programming_language_origin = "Python"
        programming_language_dest = "C++"

        graph_structure_analyzer = FactoryGraphTranscoder().create(
            gen_ai_memory=gen_ai_memory,
            programming_language_origin=programming_language_origin,
            programming_language_dest=programming_language_dest,
            gen_ai_llm_16k_plus=gen_ai_llm_16k,
            gen_ai_llm_128k_plus=gen_ai_llm_128k,
            enum_include_structure=EnumFactoryGraphTranscoderMode.GRAPH_ONLY_STRUCTURE_ANALYZER,
        )
        print(str(graph_structure_analyzer))

        graph_transcoder = FactoryGraphTranscoder().create(
            gen_ai_memory=gen_ai_memory,
            programming_language_origin=programming_language_origin,
            programming_language_dest=programming_language_dest,
            gen_ai_llm_16k_plus=gen_ai_llm_16k,
            gen_ai_llm_128k_plus=gen_ai_llm_128k,
            enum_include_structure=EnumFactoryGraphTranscoderMode.GRAPH_ONLY_TRANSCODER,
        )
        print(str(graph_transcoder))

        graph_structure_analyzer_result = graph_structure_analyzer.run(
            {
                "code": sample_file,
            },
            user_natural_language_input="code",
        )

        structure_final_output = graph_structure_analyzer_result["final_output"]

        print(structure_final_output)

        graph_transcoder_result = graph_transcoder.run(
            {
                "code": sample_file,
                "project_or_class_summary": structure_final_output,
            },
            user_natural_language_input="code",
        )

        transcoder_final_output_documenter = graph_transcoder_result[
            "output_node_code_documenter"
        ]
        transcoder_final_output_unit_testing_generator = graph_transcoder_result[
            "output_node_code_unit_testing_generator"
        ]

        print(transcoder_final_output_documenter)
        print(transcoder_final_output_unit_testing_generator)


if mode in ("test", "prod"):
    test_graph_transcoder()
