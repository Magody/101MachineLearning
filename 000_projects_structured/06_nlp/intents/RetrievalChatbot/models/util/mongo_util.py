
import os

from ..db.MongoDatabase import *
import re

from .data_preparation import *



def getChatbotsResponses(mongoDatabase: MongoDatabase, verbose=0) -> dict:

    responses = mongoDatabase.find_all("responses")

    if verbose > 0:
        print(responses)

    responses_dict = dict()
    for response in responses:

        _id = response["_id"]
        responses_dict[_id] = dict()
        for key, value in response.items():
            if key == "_id":
                continue

            responses_dict[_id][key] = value

    if verbose > 0:
        print(responses_dict)

    return responses_dict


