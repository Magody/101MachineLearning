from distutils.dir_util import copy_tree, remove_tree
from MongoDatabase import MongoDatabase
from copy import deepcopy
import subprocess
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json

# console configuration

f = open(os.path.join(".env.json"))
data = json.load(f)

global MONGODB_USER, MONGODB_PASSWORD, HOST, MONGO_DB_PORT
HOST = data["HOST"]
MONGO_DB_PORT = data["MONGO_DB_PORT"]
MONGODB_USER = data["MONGODB_USER"]
MONGODB_PASSWORD =  data["MONGODB_PASSWORD"]
APP_PORT =  data["APP_PORT"]
f.close()


ENDPOINT_TRAIN = f"/api/v1/chatbot/retrieval/train"
ENDPOINT_BRAIN_TRANSFER = f"/api/v1/chatbot/retrieval/brain/transfer"
ENDPOINT_BRAIN_CLONE = f"/api/v1/chatbot/retrieval/brain/clone"
ENDPOINT_BRAIN_DELETE = f"/api/v1/chatbot/retrieval/brain/delete"

app = Flask(__name__)

cors = CORS(app, resources={r"/"+ENDPOINT_TRAIN: {"origins": "*"}})


@app.route(ENDPOINT_TRAIN, methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type'])
def trainSpecific():

    res = {
        "id_message": 0,
        "client_message": "begining training",
        "metadata": "",
        "data": [],
        "error": ""
    }

    content = request.json
    print(content)
    db_name = content["db_name"]
    user = content["user"]
    directory = content["directory"]
    brain_name = content["brain_name"]
    chatbot_name = content["chatbot_name"]

    try:
        # python dictionarychatbot/train.py chatbot_test test brain_telegram cb_firma -1 false
        output = subprocess.check_output(
            f"cd ../{directory} && python train.py {db_name} {user} {brain_name} {chatbot_name} -1 false && cd ..", shell=True, stderr=subprocess.STDOUT)
        # print(str(output))
        output_split = str(output).split("*****")

        client_response = output_split[1]
        print("output", client_response)
        res["id_message"] = 1
        res["client_message"] = client_response

    except subprocess.CalledProcessError as e:
        res["id_message"] = -1
        res["error"] = str(e.output).replace("\r", "").replace("\n","").replace("\\r", "").replace("\\n","").replace("Traceback (most recent call last):", "")
    except Exception as exception:
        res["id_message"] = -1
        res["error"] = str(exception)

    # response = jsonify(res)

    # Enable Access-Control-Allow-Origin
    # response.headers.add("Access-Control-Allow-Origin", "*")

    return jsonify(res)


@app.route(ENDPOINT_BRAIN_TRANSFER, methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type'])
def transferKnowledge():

    res = {
        "id_message": 0,
        "client_message": "begining transference of knowledge",
        "metadata": "",
        "data": [],
        "error": ""
    }

    content = request.json
    print(content)

    # you also can consider transfer knowledge from diferents users

    from_db_name = content["from_db_name"]
    from_user = content["from_user"]
    from_directory = content["from_directory"]
    from_brain_name = content["from_brain_name"]
    to_brain_name = content["to_brain_name"]

    try:

        global MONGODB_USER, MONGODB_PASSWORD, HOST, MONGO_DB_PORT
        mongoDatabase = MongoDatabase(
            from_db_name, "mongodb://%s:%s@%s:%d" % (MONGODB_USER, MONGODB_PASSWORD, HOST, MONGO_DB_PORT))

        from_brain = mongoDatabase.find("brains", {
            "user": from_user,
            "name": from_brain_name
        })

        to_brain = mongoDatabase.find("brains", {
            "user": from_user,
            "name": to_brain_name
        })

        # copy_brain = deepcopy(from_brain)
        # change user or other stuff... if is needed, you shouldnt change name, user or _id
        to_brain["config"]["tree_broker"] = from_brain["config"]["tree_broker"]
        to_brain["others"] = from_brain["others"]
        to_brain["chatbots"] = from_brain["chatbots"]

        # transference is update the old brain
        modified_count = mongoDatabase.update(
            "brains", {"_id": to_brain["_id"]}, to_brain)
        if modified_count == 0:
            raise Exception(
                "No se registraron cambios, los dos cerebros son iguales")

        # in the same folder
        from_folder = f"../{from_directory}/chatbots/models_saved/{from_brain_name}"
        to_folder = f"../{from_directory}/chatbots/models_saved/{to_brain_name}"

        copy_tree(from_folder, to_folder)

        res["id_message"] = 1
        res["client_message"] = "Transferido todo exitosamente"

    except Exception as exception:
        res["id_message"] = -1
        res["error"] = str(exception)

    # response = jsonify(res)

    # Enable Access-Control-Allow-Origin
    # response.headers.add("Access-Control-Allow-Origin", "*")

    return jsonify(res)


@app.route(ENDPOINT_BRAIN_CLONE, methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type'])
def cloneBrain():

    res = {
        "id_message": 0,
        "client_message": "begining clonation of brain",
        "metadata": "",
        "data": [],
        "error": ""
    }

    content = request.json
    print(content)

    from_db_name = content["from_db_name"]
    from_user = content["from_user"]
    from_directory = content["from_directory"]
    from_brain_name = content["from_brain_name"]
    to_brain_name = content["to_brain_name"]

    try:

        global MONGODB_USER, MONGODB_PASSWORD, HOST, MONGO_DB_PORT
        mongoDatabase = MongoDatabase(
            from_db_name, "mongodb://%s:%s@%s:%d" % (MONGODB_USER, MONGODB_PASSWORD, HOST, MONGO_DB_PORT))

        from_brain = mongoDatabase.find("brains", {
            "user": from_user,
            "name": from_brain_name
        })

        new_brain = deepcopy(from_brain)
        new_brain["_id"] = from_user + "_" + to_brain_name
        new_brain["user"] = from_user
        new_brain["name"] = to_brain_name

        # transference is update the old brain
        result = mongoDatabase.insertDocument("brains", new_brain)
        if result == "":
            raise Exception(
                "Error, no se pudo guardar el dato en la base de datos")

        # in the same folder or it can be in diferent folder
        from_folder = f"../{from_directory}/chatbots/models_saved/{from_brain_name}"
        to_folder = f"../{from_directory}/chatbots/models_saved/{to_brain_name}"

        copy_tree(from_folder, to_folder)

        res["id_message"] = 1
        res["client_message"] = "Clonado todo exitosamente"

    except Exception as exception:
        res["id_message"] = -1
        res["error"] = str(exception)

    # response = jsonify(res)

    # Enable Access-Control-Allow-Origin
    # response.headers.add("Access-Control-Allow-Origin", "*")

    return jsonify(res)


@app.route(ENDPOINT_BRAIN_DELETE, methods=['DELETE'])
@cross_origin(origin='*', headers=['Content-Type'])
def deleteBrain():

    res = {
        "id_message": 0,
        "client_message": "begining delete brain",
        "metadata": "",
        "data": [],
        "error": ""
    }

    content = request.json
    print(content)

    db_name = content["db_name"]
    user = content["user"]
    directory = content["directory"]
    brain_name = content["brain_name"]

    try:

        global MONGODB_USER, MONGODB_PASSWORD, HOST, MONGO_DB_PORT
        mongoDatabase = MongoDatabase(
            db_name, "mongodb://%s:%s@%s:%d" % (MONGODB_USER, MONGODB_PASSWORD, HOST, MONGO_DB_PORT))

        deleted_count = mongoDatabase.delete("brains", {
            "user": user,
            "name": brain_name
        })

        if deleted_count == 0:
            raise Exception("No se pudo borrar el cerebro de la base de datos")

        folder = f"../{directory}/chatbots/models_saved/{brain_name}"

        remove_tree(folder)

        res["id_message"] = 1
        res["client_message"] = "Eliminado exitosamente"

    except Exception as exception:
        res["id_message"] = -1
        res["error"] = str(exception)

    # response = jsonify(res)

    # Enable Access-Control-Allow-Origin
    # response.headers.add("Access-Control-Allow-Origin", "*")

    return jsonify(res)


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=APP_PORT, debug=APP_PORT==7000)

    except Exception as error:
        print("error", error)
    finally:
        print("end program")
