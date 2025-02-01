import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
import json
from ai_framework.ConversationDBHandler import ConversationDBHandler


class DynamoDBHandler(ConversationDBHandler):
    """
    A class that provides a wrapper around the AWS DynamoDB client to simplify interactions with DynamoDB tables.

    Attributes:
        type_serializer (TypeSerializer): An instance of the TypeSerializer class from the boto3 library, used to serialize Python objects to DynamoDB data types.
        type_deserializer (TypeDeserializer): An instance of the TypeDeserializer class from the boto3 library, used to deserialize DynamoDB data types to Python objects.
        conn (boto3.client): An instance of the DynamoDB client from the boto3 library, used to interact with DynamoDB.
    """

    def __init__(
        self,
        config={
            "region_name": "us-east-1",
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
        },
    ):
        """
        Initializes the DynamoDBHandler class.

        Args:
            config (dict, optional): A dictionary containing the AWS region name, access key ID, and secret access key. Defaults to a dictionary with the region name set to "us-east-1" and empty access key ID and secret access key.
        """
        super().__init__(config)
        self.type_serializer = TypeSerializer()
        self.type_deserializer = TypeDeserializer()
        self.conn = boto3.client(
            "dynamodb",
            region_name=config["region_name"],
            aws_access_key_id=config["aws_access_key_id"],
            aws_secret_access_key=config["aws_secret_access_key"],
        )

    def dynamodb_item_to_dict(self, item: dict):
        """
        Converts a DynamoDB item (represented as a dictionary) to a Python dictionary.

        Args:
            item (dict): A DynamoDB item represented as a dictionary.

        Returns:
            dict: A Python dictionary representing the DynamoDB item.
        """
        return {
            key: self.type_deserializer.deserialize(value)
            for key, value in item.items()
        }

    def dict_to_dynamodb_item(self, python_dict: dict):
        """
        Converts a Python dictionary to a DynamoDB item (represented as a dictionary).

        Args:
            python_dict (dict): A Python dictionary to be converted to a DynamoDB item.

        Returns:
            dict: A DynamoDB item represented as a dictionary.
        """
        item_string = json.dumps(python_dict, indent=4, sort_keys=True, default=str)

        return {
            key: self.type_serializer.serialize(value)
            for key, value in json.loads(item_string).items()
        }

    def put(self, table_name, item_as_dict: dict):
        """
        Writes a new item to a DynamoDB table.

        Args:
            table_name (str): The name of the DynamoDB table.
            item_as_dict (dict): A Python dictionary representing the item to be written to the table.

        Returns:
            dict: The response from the DynamoDB client after the item is written.
        """
        item = self.dict_to_dynamodb_item(item_as_dict)
        response = self.conn.put_item(TableName=table_name, Item=item)
        return response

    def get(self, table_name, key_as_dict: dict):
        """
        Retrieves an item from a DynamoDB table based on the provided key.

        Args:
            table_name (str): The name of the DynamoDB table.
            key_as_dict (dict): A Python dictionary representing the key of the item to be retrieved.

        Returns:
            dict: The response from the DynamoDB client containing the retrieved item.
        """
        item_key = self.dict_to_dynamodb_item(key_as_dict)
        response = self.conn.get_item(TableName=table_name, Key=item_key)
        return response
