
import requests
import json

class APIWaliChat:

    API_URL = "https://api.wali.chat/v1"
    ENDPOINT_MESSAGES = "/messages"
    ENDPOINT_IO = "/io"
    ENDPOINT_CHATS = "/chats"
    ENDPOINT_NOTES = "/notes"
    ENDPOINT_LABELS = "/labels"
    ENDPOINT_OWNER = "/owner"
    ENDPOINT_STATUS = "/status"

    STATUS_REMOVED = "removed" 
    STATUS_ARCHIVED = "archived"
    STATUS_PENDING = "pending"
    STATUS_ACTIVE = "active"
    STATUS_RESOLVED = "resolved"

    

    def __init__(self, token: str):
        self.headers = {
            "Content-Type": "application/json",
            "Token": token
        }


    def sendMessage(self, phone_number: str, message: str):
        url = f"{APIWaliChat.API_URL}{APIWaliChat.ENDPOINT_MESSAGES}"
        
        object_message = {
            'phone': phone_number,
            "message": message
            }

        return requests.post(url, json = object_message, headers = self.headers)

    def sendMessageMultimedia(self, phone_number: str, message: str, fileID: str):
        url = f"{APIWaliChat.API_URL}{APIWaliChat.ENDPOINT_MESSAGES}"

        object_message = {
            'phone': phone_number,
            "message": message,
            "media": {
                "file": fileID
            }
        }

        return requests.post(url, json = object_message, headers = self.headers)

    def createNote(self, device_id: str, chatWId: str, note: str):
        url = f"{APIWaliChat.API_URL}{APIWaliChat.ENDPOINT_IO}/{device_id}{APIWaliChat.ENDPOINT_CHATS}/{chatWId}{APIWaliChat.ENDPOINT_NOTES}"

        print(url)

        object_message = {
            "message": "Esta es una nota interna solo es visible para el agente:\n" + note
        }

        return requests.post(url, json = object_message, headers = self.headers)

    def assignLabels(self, device_id: str, chatWId: str, labels: list = ["bot_error"]):
        # if empty, delete all labels
        url = f"{APIWaliChat.API_URL}{APIWaliChat.ENDPOINT_IO}/{device_id}{APIWaliChat.ENDPOINT_CHATS}/{chatWId}{APIWaliChat.ENDPOINT_LABELS}"


        object_message = labels

        return requests.patch(url, json = object_message, headers = self.headers)

    def assignAgent(self, device_id: str, chatWId: str, agent: dict):
        url = f"{APIWaliChat.API_URL}{APIWaliChat.ENDPOINT_IO}/{device_id}{APIWaliChat.ENDPOINT_CHATS}/{chatWId}{APIWaliChat.ENDPOINT_OWNER}"

        object_message = agent

        return requests.patch(url, json = object_message, headers = self.headers)

    def assignStatus(self, device_id: str, chatWId: str, status: str):
        url = f"{APIWaliChat.API_URL}{APIWaliChat.ENDPOINT_IO}/{device_id}{APIWaliChat.ENDPOINT_CHATS}/{chatWId}{APIWaliChat.ENDPOINT_STATUS}"

        object_message = {
            "status": status
        }

        return requests.patch(url, json = object_message, headers = self.headers)

    
    def getChatDetails(self, device_id: str, chatWId: str):
        url = f"{APIWaliChat.API_URL}{APIWaliChat.ENDPOINT_IO}/{device_id}{APIWaliChat.ENDPOINT_CHATS}/{chatWId}"

        return requests.get(url, headers = self.headers)
        
    
    

    


        

    