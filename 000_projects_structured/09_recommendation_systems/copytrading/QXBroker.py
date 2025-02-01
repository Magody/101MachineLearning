import os
import time
import datetime
import asyncio
from pathlib import Path
from termcolor import colored
from quotexpy import Quotex
from quotexpy.utils import asset_parse
from quotexpy.utils.account_type import AccountType
from quotexpy.utils.operation_type import OperationType
from quotexpy.utils.duration_time import DurationTime
from enum import Enum
from pytz import timezone

class EnumLoggingLevels(Enum):
    NOTHING = 0
    INFO = 1
    ERROR = 2
    WARNING = 3
    DEBUG = 4


class TimeHandler:
    """IMPORTANT: Every transformed time is for UTC 0"""

    @staticmethod
    def get_now_utc(utc_offset: int=0):
        
        return datetime.datetime.now(timezone("UTC")) + datetime.timedelta(hours=utc_offset)
    
    @staticmethod
    def parse_hour_minute(utc_offset, time_str):
        hour, minute = map(int, time_str.split(':'))

        current_time = TimeHandler.get_now_utc()
        parsed_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        local_time = parsed_time + datetime.timedelta(hours=utc_offset*-1)  # Reverse de UTC to estabilish to 0
        return local_time

class QXBroker:
    """
    This class represents a broker connection object and provides methods for various trading operations.
    """

    def __init__(
            self,
            account_type: AccountType,
            email: str,
            password:str, 
            utc_offset: int,
            headless=False, 
            verbose_level: EnumLoggingLevels=EnumLoggingLevels.INFO
        ):
        self.account_type = account_type
        self.email = email
        self.headless = headless

        self.client = Quotex(email=self.email, password=password, headless=self.headless)
        self.client.debug_ws_enable = False
        self.verbose_level = verbose_level


    def change_account(self, account_type: AccountType=None):

        if account_type is not None:
            self.account_type = account_type
        self.client.change_account(self.account_type)

    def log(self, message: str, verbose_min: EnumLoggingLevels=EnumLoggingLevels.INFO):
        if self.verbose_level.value >= verbose_min.value:
            
            now = TimeHandler.get_now_utc()
            message = f"{now}: {message}"

            color = "white"
            if verbose_min == EnumLoggingLevels.INFO:
                color = "blue"
                message = f"[INFO] {message}"
            elif verbose_min == EnumLoggingLevels.ERROR:
                color = "red"
                message = f"[ERROR] {message}"
            elif verbose_min == EnumLoggingLevels.WARNING:
                color = "yellow"
                message = f"[WARNING] {message}"
            elif verbose_min == EnumLoggingLevels.DEBUG:
                color = "cyan"
                message = f"[DEBUG] {message}"

            print(colored(message, color))

    async def connect(self, attempts=5):
        check, reason = await self.client.connect()
        if not check:
            attempt = 0
            while attempt <= attempts:
                if not self.client.check_connect():
                    check, reason = await self.client.connect()
                    if check:
                        self.log("Conexión exitosa")
                        break
                    self.log("Error al conectar", EnumLoggingLevels.ERROR)
                    attempt += 1
                    if Path(os.path.join(".", "session.json")).is_file():
                        Path(os.path.join(".", "session.json")).unlink()
                    self.log(f"Reintentando conexión: intento {attempt} de {attempts}")
                elif not check:
                    attempt += 1
                else:
                    break
                await asyncio.sleep(5)
            return check, reason
        return check, reason

    def close(self):
        """
        Closes the client connection.
        """
        self.client.close()

    async def check_asset(self, asset):
        check_connect, message = await self.connect()
        if check_connect:
            asset_query = asset_parse(asset)
            asset_open = self.client.check_asset_open(asset_query)

            try:
                try_otc = False
                if asset_open:
                    if not asset_open[2]:
                        self.log("Asset is closed.", EnumLoggingLevels.WARNING)
                        try_otc = True
                else:
                    self.log(f"Asset not found, try another.", EnumLoggingLevels.WARNING)
                    try_otc = True

                if try_otc:
                    asset = f"{asset}_otc"
                    self.log(f"Trying OTC Asset -> {asset}", EnumLoggingLevels.WARNING)
                    asset_query = asset_parse(asset)
                    asset_open = self.client.check_asset_open(asset_query)

                    if not asset_open:
                        self.log(f"Asset not found, try another. Probably we can't access to that asset.", EnumLoggingLevels.WARNING)

            except Exception as error:
                self.log(f"Error: {error}. self.client.api.instruments={self.client.api.instruments}", EnumLoggingLevels.ERROR)

            return asset, asset_open
        return asset, None

    async def get_balance(self):
        check_connect, _ = await self.connect()
        balance = None
        if check_connect:
            self.change_account()
            balance = await self.client.get_balance()
            self.log(f"Balance: {balance}", EnumLoggingLevels.DEBUG)
        self.close()
        return balance

    async def balance_refill(self, force=False):
        if self.account_type == AccountType.PRACTICE or force:
            check_connect, _ = await self.connect()
            if check_connect:
                result = await self.client.edit_practice_balance(100)
                self.log(str(result), EnumLoggingLevels.DEBUG)
            self.close()
        else:
            self.log("You are trying to refil a non PRACTICE account, check if you really want to do this", EnumLoggingLevels.WARNING)

    async def trade(self, asset: str, action: OperationType, amount: int, duration_seconds: DurationTime):
        check_connect, _ = await self.connect()
        trade_status, trade_info = None, None
        if check_connect:
            self.change_account()
            asset_found, asset_open = await self.check_asset(asset)
            
            if asset_open:
                if asset_open[2]:
                    self.log(f"Asset {asset_found} is open.", EnumLoggingLevels.INFO)
                    try:
                        trade_status, trade_info = await self.client.trade(action, amount, asset_found, duration_seconds)
                    except Exception as error:
                        self.log(f"Error while trading: {error}", EnumLoggingLevels.ERROR)
                else:
                    self.log(f"Asset is closed. Can't trade", EnumLoggingLevels.WARNING)
            else:
                self.log(f"Asset is closed. Can't trade", EnumLoggingLevels.WARNING)

        self.close()
        return asset_found, trade_status, trade_info

    async def wait_until_utc_time(self, datetime_to_wait: datetime.datetime):
        while True:
            if TimeHandler.get_now_utc() >= datetime_to_wait:
                return  # Returns when it's the right time to proceed
            await asyncio.sleep(0.5)

    async def check_profit(self, datetime_to_wait: datetime.datetime, asset, trade_status, trade_info):
        await self.client.check_win(asset, trade_info["id"])
        return None

    async def sell_option(self, trade_status, trade_info):
        check_connect, _ = await self.connect()
        result = None
        if check_connect:
            self.change_account()
            if trade_status:
                result = await self.client.sell_option(trade_info["id"])
            else:
                self.log(f"Not Traded {trade_info}", EnumLoggingLevels.INFO)
        self.close()
        return result

    async def assets_open(self):
        check_connect, _ = await self.connect()
        assets = []
        if check_connect:
            self.change_account()
            for i in self.client.get_all_asset_name():
                assets.append([i, self.client.check_asset_open(i)])
        self.close()
        return assets

    async def get_payment(self):
        check_connect, _ = await self.connect()
        if check_connect:
            self.change_account()
            all_data = self.client.get_payment()
            for asset_name in all_data:
                asset_data = all_data[asset_name]
                print(asset_name, asset_data["payment"], asset_data["open"])
        self.close()

    async def get_candle_v2(self):
        check_connect, message = await self.connect()
        period = 100
        if check_connect:
            asset, asset_open = await self.check_asset(self.asset_current)
            if asset_open[2]:
                print(colored("[INFO]: ", "blue"), "Asset is open.")
                candles = await self.client.get_candle_v2(asset, period)
                print(candles)
            else:
                print(colored("[INFO]: ", "blue"), "Asset is closed.")
        self.close()

    async def get_realtime_candle(self):
        check_connect, message = await self.connect()
        if check_connect:
            list_size = 10
            asset, asset_open = await self.check_asset(self.asset_current)
            self.client.start_candles_stream(asset, list_size)
            while True:
                if len(self.client.get_realtime_candles(asset)) == list_size:
                    break
            print(self.client.get_realtime_candles(asset))
        self.close()

    async def get_signal_data(self):
        check_connect, message = await self.connect()
        if check_connect:
            while True:
                print(self.client.get_signal_data())
                time.sleep(1)
        self.close()
