import os
import sys

import pandas as pd
from ai_framework.GenAILLM import EnumGenAIModelsIdsBedrock, EnumGenAIPlatforms, GenAILLM
from ai_framework.GenAIMemory import GenAIMemory
from ai_framework.LoggingAndTelemetry import EnumLogs
from ai_framework.prefabs.code.PrefabNodeCleanCode import PrefabNodeCleanCode
from ai_framework.prefabs.code.PrefabNodeCodeDocumenter import PrefabNodeCodeDocumenter
from ai_framework.prefabs.code.PrefabNodeCodeUnitTester import PrefabNodeCodeUnitTester
from ai_framework.prefabs.code.PrefabNodeStructureAnalizer import PrefabNodeCodeStructureAnalizer
from ai_framework.prefabs.code.PrefabNodeTranscoder import PrefabNodeTranscoder

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

        prefab_node_clean = PrefabNodeCleanCode("node_next")
        node_clean = prefab_node_clean.node

        node_clean.invoke({'code': 'abc(){ }\n\n\n\nABC'}, gen_ai_memory)

        prefab_node_structure_analizer = PrefabNodeCodeStructureAnalizer(
            gen_ai_llm_16k,
            "Python",
            "node_next"
        )
        node_structure_analizer = prefab_node_structure_analizer.node

        res = node_structure_analizer.invoke({'code': sample_file}, gen_ai_memory)
        print(res)
        print(res["final_output"])

        prefab_node_transcoder = PrefabNodeTranscoder(
            gen_ai_llm_16k,
            programming_language_origin="Python",
            programming_language_dest="Cobol",
            project_or_class_summary="",
            node_id_next="node_next"
        )

        node_transcoder = prefab_node_transcoder.node

        res = node_transcoder.invoke({'code': sample_file}, gen_ai_memory)
        print(res)
        print(res["final_output"])

        prefab_node_code_documenter = PrefabNodeCodeDocumenter(
            gen_ai_llm_16k,
            programming_language_to_doc="Python",
            node_id_next="node_next"
        )

        node_code_documenter = prefab_node_code_documenter.node

        res = node_code_documenter.invoke({'code': sample_file}, gen_ai_memory)
        print(res)
        print(res["final_output"])

        prefab_node_code_unit_tester = PrefabNodeCodeUnitTester(
            gen_ai_llm_16k,
            programming_language_for_unit_testing="Python",
            project_or_class_summary="",
            node_id_next="node_next"
        )

        node_code_unit_tester = prefab_node_code_unit_tester.node

        res = node_code_unit_tester.invoke({'code': sample_file}, gen_ai_memory)
        print(res)
        print(res["final_output"])


if mode in ("test", "prod"):
    test_graph_transcoder()
