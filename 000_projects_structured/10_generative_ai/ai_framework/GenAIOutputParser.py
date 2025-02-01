from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from typing import List
import json
from enum import Enum
import xmltodict


class EnumCommonParsers(Enum):
    """
    An enumeration of common parsers used in the GenAIOutputParser class.
    """
    PARSER_REASONING_ANSWER = "reasoning-answer"


class GenAIOutputParser:
    """
    A class that provides functionality for parsing the output of a GenAI (Generative AI) model.

    Attributes:
        use_langchain (bool): A flag indicating whether to use the LangChain library for parsing the output.
        natural_language_key (str): The key used to access the natural language response in the parsed output.
        response_schemas (List[dict]): A list of dictionaries representing the response schemas.
        output_parser (StructuredOutputParser): The LangChain output parser, if `use_langchain` is True.
        format_instructions (str): The instructions for formatting the output.

    Methods:
        get_instructions(): Returns the format instructions for the output.
        parse(output): Parses the output and returns the result.
        create_common_parser(common_parser: EnumCommonParsers): Creates a common parser based on the provided EnumCommonParsers value.
    """

    def __init__(
        self,
        response_schemas: List[dict],
        use_langchain=False,
        natural_language_key="respuesta",
    ) -> None:
        """
        Initializes the GenAIOutputParser object.

        Args:
            response_schemas (List[dict]): A list of dictionaries representing the response schemas.
            use_langchain (bool, optional): A flag indicating whether to use the LangChain library for parsing the output. Defaults to False.
            natural_language_key (str, optional): The key used to access the natural language response in the parsed output. Defaults to "respuesta".
        """
        self.use_langchain = use_langchain
        self.natural_language_key = natural_language_key

        if use_langchain:
            self.response_schemas = []
            for schema in response_schemas:
                self.response_schemas.append(
                    ResponseSchema(
                        name=schema["name"], description=schema["description"]
                    )
                )
            self.output_parser = StructuredOutputParser.from_response_schemas(
                self.response_schemas
            )
        else:
            self.response_schemas = response_schemas

        self.get_instructions()

    def get_instructions(self):
        """
        Generates the format instructions for the output.

        Returns:
            str: The format instructions for the output.
        """
        if self.use_langchain:
            self.format_instructions = (
                self.output_parser.get_format_instructions()
                .replace("\n", " ")
                .replace("\t", "")
            )
        else:
            self.format_instructions = "Devuelve la respuesta en un XML con el siguiente formato XML <final-answer>"
            for schema in self.response_schemas:
                name, description = schema["name"], schema["description"]
                self.format_instructions += (
                    f"<{name}>#TU RESPUESTA va aquí. {description}</{name}>"
                )
            self.format_instructions += "</final-answer>"

        return self.format_instructions.replace(":", "#CODE_SEMICOLON#")

    def __str__(self):
        """
        Returns the format instructions for the output.

        Returns:
            str: The format instructions for the output.
        """
        return self.get_instructions()

    def parse(self, output):
        """
        Parses the output and returns the result.

        Args:
            output (str): The output to be parsed.

        Returns:
            dict: The parsed output.
        """
        try:
            if self.use_langchain:
                return json.loads(
                    output.replace("```json", "").replace("```", "").strip()
                )
            return xmltodict.parse(output)
        except Exception as error:
            print(f"Error in output parse: {error}")
            return output

    @staticmethod
    def create_common_parser(common_parser: EnumCommonParsers):
        """
        Creates a common parser based on the provided EnumCommonParsers value.

        Args:
            common_parser (EnumCommonParsers): The common parser to be created.

        Returns:
            GenAIOutputParser: The created common parser.
        """
        if common_parser == EnumCommonParsers.PARSER_REASONING_ANSWER:
            return GenAIOutputParser(
                [
                    {
                        "name": "reasoning",
                        "description": "resumen del razonamiento detras de la respuesta",
                    },
                    {"name": "respuesta", "description": "respuesta a la consulta"},
                ],
                use_langchain=False,
                natural_language_key="respuesta",
            )
        raise Exception(f"Not implemented parser: {common_parser}")
