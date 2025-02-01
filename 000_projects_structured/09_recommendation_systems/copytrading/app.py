from telethon import TelegramClient, events
import datetime
import re
from QXBroker import QXBroker, EnumLoggingLevels, TimeHandler
from quotexpy.utils.account_type import AccountType
from quotexpy.utils.operation_type import OperationType
import nest_asyncio
import asyncio
nest_asyncio.apply()

pattern = re.compile(
    r'.* Zona horaria: UTC ([+-]\d+)\s*\n*'
    r'.*(\d+) minutos de caducidad\s*\n*'
    r'([A-Z]{3}\/[A-Z]{3};\d{2}:\d{2};(?:PUT|CALL) .*)\s*\n*'
    r'.*HORA HASTA LAS (\d{2}:\d{2})\s*\n*'
    r'1º GALE —>HORA HASTA LAS (\d{2}:\d{2})\s*\n*'
    r'2º GALE —>HORA HASTA LAS (\d{2}:\d{2})\s*\n*'
    r'.*Haga clic para abrir el broker\s*\n*'
    r'.*¿Aún no sabes operar\? Haga clic aquí\s*\n*'
)

from dotenv import load_dotenv
import os
import re
load_dotenv(".env", override=True)

global broker
# Example usage
async def main(params):
    OFFSET_TRADE_BOT_DELAY = 30
    OFFSET_PROFIT_RESULT = 10
    gale1_enabled = True

    global broker
    broker = QXBroker(
        account_type=AccountType.PRACTICE,  # REAL
        email=os.environ['email'],
        password=os.environ['password'],
        utc_offset=-3,
        headless=False,
        verbose_level=EnumLoggingLevels.DEBUG
    )

    maximum_amount = int(params["maximum_amount"])

    message_text = params["message_text"]

    matches = pattern.match(message_text)
    if matches:
        broker.log(f"Mensaje con formato específico recibido: {message_text}")

        groups = matches.groups()
        broker.log(f"Groups: {groups}", EnumLoggingLevels.DEBUG)

        utc_offset = int(groups[0])
        caducity_minutes = int(groups[1])
        duration_seconds = caducity_minutes * 60

        payload = groups[2].split(";")
        asset = re.sub(r'[^A-z]+', '', payload[0].upper())
        start_time = TimeHandler.parse_hour_minute(utc_offset, payload[1])
        action = OperationType.PUT_RED if ("PUT" in payload[2] or "🟥" in payload[2]) else OperationType.CALL_GREEN

        until_time_real = (
            start_time + 
            datetime.timedelta(seconds=duration_seconds)
        )
        gale_1_time = TimeHandler.parse_hour_minute(utc_offset, groups[4])  # Maximum to this time
        # gale_2_time = TimeHandler.parse_hour_minute(utc_offset, groups[5])

        broker.log(f"|asset:{asset}|start_time:{start_time}|action:{action}|", EnumLoggingLevels.DEBUG)

        broker.utc_offset = utc_offset  # Necessary update

        broker.log(f"Getting initial balance...", EnumLoggingLevels.INFO)
        initial_balance = await broker.get_balance()

        amount = min(maximum_amount, max(1, int(initial_balance * 0.02)))  # Stake 2%

        # Before trade, wait until the right moment (OFFSET_TRADE_BOT_DELAY DELAY of BOT TO CHECK ASSET...)
        until_time_uct_begin = start_time - datetime.timedelta(seconds=OFFSET_TRADE_BOT_DELAY)
        broker.log(f"WAITING UNTIL TIME UTC: {until_time_uct_begin}", EnumLoggingLevels.DEBUG)
        await broker.wait_until_utc_time(until_time_uct_begin)

        async def helper_trade(amount, duration_seconds):
            final_balance = initial_balance
            broker.log("STARTING TRADE",  EnumLoggingLevels.DEBUG)
            asset_found, trade_status, trade_info = await broker.trade(
                asset,
                action,
                amount=amount,
                duration_seconds=duration_seconds
            )

            if trade_status:
                # {'id': '1a405b53-1aae-465a-b1be-1242c24c677f', 'openTime': '2024-07-06 21:08:07', 'closeTime': '2024-07-06 21:13:07', 'openTimestamp': 1720300087, 'closeTimestamp': 1720300387, 'uid': 44477579, 'isDemo': 1, 'tournamentId': 0, 'amount': 1, 'purchaseTime': 1720300357, 'profit': 0.9, 'percentProfit': 90, 'percentLoss': 100, 'openPrice': 0.9381, 'copyTicket': '', 'closePrice': 0, 'command': 1, 'asset': 'AUDCAD_otc', 'nickname': '#44477579', 'accountBalance': 9984, 'requestId': 1720300087, 'openMs': 535, 'currency': 'USD'}
                broker.log(f"TRADE PLACED: {trade_info}", EnumLoggingLevels.INFO)
                until_time_profit = (
                    TimeHandler.get_now_utc() + 
                    datetime.timedelta(seconds=duration_seconds + OFFSET_PROFIT_RESULT)
                )
                broker.log(f"WAITING UNTIL OPERATION END: {until_time_profit}")
                await broker.wait_until_utc_time(until_time_profit)

                broker.log(f"Getting final balance...", EnumLoggingLevels.INFO)
                final_balance = await broker.get_balance()
            else:
                broker.log("Not traded", EnumLoggingLevels.WARNING)
            
            profit = final_balance - initial_balance

            return profit
        
        profit = await helper_trade(amount, duration_seconds)

        if profit > 0:
            broker.log(f"WIN -> PROFIT: {profit}", EnumLoggingLevels.INFO)
        else:
            broker.log(f"LOSE -> PROFIT: {profit}", EnumLoggingLevels.INFO)

            if gale1_enabled:
                # Another try
                await broker.wait_until_utc_time(until_time_real)
                profit_gale = await helper_trade(amount * 2, duration_seconds - OFFSET_PROFIT_RESULT)
                broker.log(f"GALE PROFIT: {profit_gale}", EnumLoggingLevels.INFO)

        broker.log(f"REQUEST END", EnumLoggingLevels.DEBUG)

    else:
        broker.log(f"Mensaje recibido pero no coincide con el formato específico: <|{message_text}|>", EnumLoggingLevels.INFO)
    

def run(y):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    z = loop.run_until_complete(y)
    return z

# Reemplaza estos valores con los obtenidos de https://my.telegram.org/
api_id = '27943863'
api_hash = '7ec8710ab54e48aac932a99397b7a635'
channel_username = "magictradersignals"  # Reemplaza con el nombre de usuario del canal. -2164829186
# +o9dPfb68BFliY2Fh magictradersignals
# Crear el cliente de Telethon
client = TelegramClient('session_name', api_id, api_hash)

@client.on(events.NewMessage(chats=channel_username))
async def handler(event):
    # Aquí puedes manejar el mensaje recibido
    message_text = event.message.text
    try:

        matches = pattern.match(message_text)
        if matches:
            print("OK")
            run(main({
                "message_text": message_text,
                "maximum_amount": 200
            }))
            """
            """
        else:
            print(f"Mensaje recibido pero no coincide con el formato específico: <|{message_text}|>")
            # pass

    except Exception as error:
        print(error)

# Iniciar el cliente
client.start()
print(f"Escuchando mensajes en el canal: {channel_username}.")  # await client.get_me()

# Mantener el script en ejecución
client.run_until_disconnected()
