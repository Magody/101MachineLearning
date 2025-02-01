from datetime import datetime
import pytz
from enum import Enum


class EnumLogs(Enum):
    LOG_LEVEL_NOTHING = 0
    LOG_LEVEL_INFO = 1
    LOG_LEVEL_WARNING = 2
    LOG_LEVEL_ERROR = 3
    LOG_LEVEL_DEBUG = 4


class Logging:

    @staticmethod
    def log(
        message,
        verbose_level: EnumLogs,
        minimum_verbose_level: EnumLogs = EnumLogs.LOG_LEVEL_INFO,
        metadata={},
    ):
        if verbose_level.value < minimum_verbose_level.value:
            return

        today = datetime.now(pytz.timezone("America/Guayaquil"))
        print(f"{today}: {message}")
