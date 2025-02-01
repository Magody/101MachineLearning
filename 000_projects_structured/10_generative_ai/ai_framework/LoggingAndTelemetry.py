from datetime import datetime
import pytz
from enum import Enum


class EnumLogs(Enum):
    """
    An enumeration class that defines the different log levels.

    Attributes:
        LOG_LEVEL_NOTHING (int): Represents no logging.
        LOG_LEVEL_INFO (int): Represents informational logging.
        LOG_LEVEL_WARNING (int): Represents warning logging.
        LOG_LEVEL_ERROR (int): Represents error logging.
        LOG_LEVEL_DEBUG (int): Represents debug logging.
    """
    LOG_LEVEL_NOTHING = 0
    LOG_LEVEL_INFO = 1
    LOG_LEVEL_WARNING = 2
    LOG_LEVEL_ERROR = 3
    LOG_LEVEL_DEBUG = 4


class Logging:
    """
    A class that provides logging functionality.

    Methods:
        log(message, verbose_level, minimum_verbose_level=EnumLogs.LOG_LEVEL_INFO, metadata={}):
            Logs a message with the specified verbose level and minimum verbose level.

            Args:
                message (str): The message to be logged.
                verbose_level (EnumLogs): The verbose level of the message.
                minimum_verbose_level (EnumLogs, optional): The minimum verbose level to log. Defaults to EnumLogs.LOG_LEVEL_INFO.
                metadata (dict, optional): Additional metadata to be included in the log. Defaults to an empty dictionary.

            Returns:
                None
    """

    @staticmethod
    def log(
        message,
        verbose_level: EnumLogs,
        minimum_verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        metadata={},
    ):
        """
        Logs a message with the specified verbose level and minimum verbose level.

        Args:
            message (str): The message to be logged.
            verbose_level (EnumLogs): The verbose level of the message.
            minimum_verbose_level (EnumLogs, optional): The minimum verbose level to log. Defaults to EnumLogs.LOG_LEVEL_INFO.
            metadata (dict, optional): Additional metadata to be included in the log. Defaults to an empty dictionary.

        Returns:
            None
        """
        if verbose_level.value < minimum_verbose_level.value:
            return

        color = ""

        today = datetime.now(pytz.timezone("America/Guayaquil"))

        if minimum_verbose_level.value == EnumLogs.LOG_LEVEL_INFO.value:
            color = "\033[1;34;40m "
        elif minimum_verbose_level.value == EnumLogs.LOG_LEVEL_WARNING.value:
            color = "\033[1;33;40m "
        elif minimum_verbose_level.value == EnumLogs.LOG_LEVEL_ERROR.value:
            color = "\033[1;31;40m "
        elif minimum_verbose_level.value == EnumLogs.LOG_LEVEL_DEBUG.value:
            color = "\033[1;35;40m "

        print(f"{color}{today} <{minimum_verbose_level}>: {message}")
