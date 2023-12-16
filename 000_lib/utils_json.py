import json

def saveJSON(path, data)->None:
    with open(path, "w") as f_out:
        json.dump(data, f_out)

def loadJSON(path)->dict:
    with open(path, "r") as f_in:
        return json.load(f_in)
    