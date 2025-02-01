from abc import ABC, abstractmethod


class ConversationDBHandler(ABC):
    """
    An abstract base class that defines the interface for handling conversations in a database.

    Attributes:
        config (dict): A dictionary containing the configuration settings for the database connection.
        conn (object): The connection object for the database.
    """

    def __init__(
        self,
        config,
    ):
        """
        Initializes the ConversationDBHandler object.

        Args:
            config (dict): A dictionary containing the configuration settings for the database connection.
        """
        self.config = config
        self.conn = None

    @abstractmethod
    def put(self, table_name):
        """
        Saves a conversation to the database.

        Args:
            table_name (str): The name of the table to save the conversation to.

        Returns:
            None
        """
        return None

    @abstractmethod
    def get(self, table_name):
        """
        Retrieves a conversation from the database.

        Args:
            table_name (str): The name of the table to retrieve the conversation from.

        Returns:
            None
        """
        return None
