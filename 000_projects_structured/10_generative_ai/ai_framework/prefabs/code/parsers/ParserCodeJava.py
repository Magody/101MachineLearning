import javalang

from ai_framework.prefabs.code.parsers.BaseParser import BaseParser


class ParserCodeJava(BaseParser):
    """
    A class that parses Java code and extracts its structure, including classes, fields, and methods.

    Attributes:
        code (str): The Java code to be parsed.
        tree (javalang.tree.CompilationUnit): The abstract syntax tree (AST) of the Java code.
        package (str): The package name of the Java code.
        imports (str): The import statements of the Java code.
        classes (list): A list of dictionaries representing the classes in the Java code, including their fields and methods.
    """

    def __init__(self, code):
        """
        Initializes the ParserCodeJava object with the given Java code.

        Args:
            code (str): The Java code to be parsed.
        """
        super().__init__(code)

    def extract_tree(self):
        """
        Extracts the abstract syntax tree (AST) of the Java code and populates the class attributes.

        This method parses the Java code using the javalang library, and then extracts the package name, import statements,
        and information about the classes, fields, and methods in the code.
        """
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
        """
        Extracts the import statements from the Java code.

        Returns:
            str: A string containing the import statements.
        """
        output = ""
        for import_path in self.tree.imports:
            output += f"import {import_path.path};\n"
        return output

    def __get_start_end_for_node(self, node_to_find):
        """
        Finds the start and end positions of a given node in the Java code.

        Args:
            node_to_find (javalang.tree.Node): The node for which to find the start and end positions.

        Returns:
            tuple: A tuple containing the start and end positions of the node.
        """
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
        """
        Extracts the string representation of a given code segment.

        Args:
            start (javalang.tree.Position): The start position of the code segment.
            end (javalang.tree.Position): The end position of the code segment.

        Returns:
            str: The string representation of the code segment.
        """
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
