from abc import ABC, abstractmethod


class BaseParser(ABC):
    """
    An abstract base class for parsing code files.

    This class provides a basic structure for parsing code files and extracting
    relevant information, such as the abstract syntax tree, package, imports,
    and classes.

    Attributes:
        code (str): The code to be parsed.
        tree (object): The extracted abstract syntax tree.
        package (str): The package of the code.
        imports (list): The list of imported modules.
        classes (list): The list of classes defined in the code.
    """

    def __init__(self, code):
        """
        Initializes the BaseParser object.

        Args:
            code (str): The code to be parsed.
        """
        self.code = code
        self.extract_tree()

    @abstractmethod
    def extract_tree(self):
        """
        Extracts the abstract syntax tree from the code.

        This method must be implemented by subclasses to extract the abstract
        syntax tree from the code. The extracted tree, package, imports, and
        classes should be stored in the corresponding attributes.
        """
        self.tree = None
        self.package = None
        self.imports = None
        self.classes = []
