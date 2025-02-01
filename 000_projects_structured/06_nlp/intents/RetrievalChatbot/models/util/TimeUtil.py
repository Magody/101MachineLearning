from datetime import datetime
from typing import Tuple

import pytz

from ..Config import Config


import os
os.environ['TZ'] = Config.TIMEZONE

# timezone = pytz.timezone(Config.TIMEZONE)



class TimeUtil:

    @staticmethod
    def getDifferenceBetweenDatetimes(datetime1: datetime, datetime2: datetime) -> Tuple[int, int, int]:
        # datetime1 > datetime2
        diff = datetime1 - datetime2

        days, seconds = diff.days, diff.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return (hours, minutes, seconds)

    @staticmethod
    def convertStringToDatetime(date_time, format = "%Y-%m-%d %H:%M:%S"):
        # The format
        return datetime.strptime(date_time, format)

    @staticmethod
    def isLaboralTime(verbose=0):

        # 8:30 a 18:00 lunes a viernes

        now = datetime.now()
        now_with_timezone = now

        # Monday is 0 and Sunday is 6.
        weekday = now_with_timezone.weekday()
        hour = now_with_timezone.time().hour
        minute = now_with_timezone.time().minute

        if weekday >= 0 and weekday <= 4:
            
            if verbose > 0:
                print("Actually is", weekday, hour, minute)

            if hour >= 8 and hour <= 18:
                if hour == 8:
                    return minute >= 30
                else:
                    return True

        return False

    @staticmethod
    def getNowWithTimeZone() -> str:
        return datetime.now()

    @staticmethod
    def getNowStringWithTimeZone() -> str:
        now_with_timezone = TimeUtil.getNowWithTimeZone()
        return now_with_timezone.strftime("%Y-%m-%d %H:%M:%S")


