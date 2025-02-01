import pymongo

class MongoDatabase:

    def __init__(self, db_name, client_direction="mongodb://localhost:27017"):

        self.client_connection = pymongo.MongoClient(client_direction)
        self.db = self.client_connection[db_name]
            
    def insertDocument(self, collection_name: str, document: dict) -> str:
        """
        return the id of the document or empty string in error
        """
        try:
            inserted = self.db[collection_name].insert_one(document)
            return inserted.inserted_id
        except Exception as error:
            print("ERROR al insertar: ", error)
            return ""

    def find(self, collection_name: str, query: dict = {}) -> dict:

        document = self.db[collection_name].find_one(query)
        return document

    def find_all(self, collection_name: str, condition:dict = {}) -> list:
        cursor = self.db[collection_name].find(condition)
        documents = []
        for document in cursor:
            documents.append(document)
        return documents

    def update(self, collection_name: str, query: dict, changes: dict) -> int:
        """
        Return count of updates
        """
        result = self.db[collection_name].update_one(query, { "$set": changes })

        return result.modified_count