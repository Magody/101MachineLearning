from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from typing import List
import json
from enum import Enum

class EnumCommonParsers(Enum):
    PARSER_REASONING_ANSWER = "reasoning-answer"

class GenAIOutputParser:

    def __init__(self, response_schemas:List[dict]) -> None:
        self.response_schemas = []
        for schema in response_schemas:
            self.response_schemas.append(ResponseSchema(
                name=schema["name"],
                description=schema["description"]
            ))

        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.get_instructions()


    @staticmethod
    def create_common_parser(common_parser:EnumCommonParsers):

        if common_parser == EnumCommonParsers.PARSER_REASONING_ANSWER:
            return GenAIOutputParser([
                {
                    "name": "reasoning",
                    "description": "resumen del razonamiento detras de la respuesta"
                },
                {
                    "name": "respuesta",
                    "description": "respuesta final concreta y concisa"
                },
            ])
        raise Exception(f"Not implemented parser: {common_parser}")

        
    def get_instructions(self):
        self.format_instructions = self.output_parser.get_format_instructions()
        return self.format_instructions
    
    def parse(self, output):
        return json.loads(output.replace("```json", "").replace("```", "").strip())