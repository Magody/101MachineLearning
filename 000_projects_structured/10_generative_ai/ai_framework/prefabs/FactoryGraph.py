from abc import ABC, abstractmethod

from ai_framework.GenAIMemory import GenAIMemory


class FactoryGraph(ABC):
    """
    An abstract base class that defines the interface for creating a graph object.

    This class provides a common interface for creating different types of graph objects,
    such as directed graphs, undirected graphs, or weighted graphs. Concrete subclasses
    of this class must implement the `create` method to create a specific type of graph
    object.
    """

    @abstractmethod
    def create(gen_ai_memory: GenAIMemory):
        """
        Create a graph object based on the provided GenAIMemory instance.

        Args:
            gen_ai_memory (GenAIMemory): An instance of GenAIMemory that contains the
                data to be used for creating the graph.

        Returns:
            The created graph object.
        """
        # Define your own solution
        pass
