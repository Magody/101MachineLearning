import os
import json

class Config:

    MODE_TEST = "test"
    MODE_STAGING = "staging"
    MODE_PRODUCTION = "production"
    MODE_GOSSIP = "gossip"

    TIMEZONE = "America/Guayaquil"

    models_saved = os.path.join("chatbots", "models_saved")

    mode = "unknown"

    TIME_SECONDS_WAIT_BASE_THREAD = 14  # in seconds

    TIME_SECONDS_WAIT_MULTIMEDIA_MESSAGE = 4
    TIME_SECONDS_WAIT_TEXT_MESSAGE = 2

    TIME_SECONDS_UPDATE_DATA = 300

    TIME_SECONDS_WAIT_FOR_FREE_MEMORY = 180
    TIME_SECONDS_WAIT_FOR_USER_MONITOR_FREE_MEMORY = 60

    TIME_MINUTES_BOT_WAIT_TO_RESET = 60

    USERNAME = "test"
    HOST = "abc.ridetel.com"
    MONGO_DB_PORT = 27018
    DB_NAME = "chatbot_test"
    MONGODB_USER = "user"
    MONGODB_PASSWORD = "password"
    APP_PORT = 5000
    SOCKET_PORT = 9000
    
    @staticmethod
    def setEnvironment(path_env_json: str):

        with open(path_env_json) as f:
            data = json.load(f)

            Config.USERNAME = data["USERNAME"]
            Config.HOST = data["HOST"]
            Config.MONGO_DB_PORT = data["MONGO_DB_PORT"]
            Config.DB_NAME = data["DB_NAME"]
            Config.MONGODB_USER = data["MONGODB_USER"]
            Config.MONGODB_PASSWORD =  data["MONGODB_PASSWORD"]
            Config.APP_PORT =  data["APP_PORT"]
            Config.SOCKET_PORT = data["SOCKET_PORT"]

            
