import math


def is_discrete(value):
    return int((value//1)*100) == int(value*100)

def is_real_number(x):
    try:
        cast_x = float(x)
        if math.isnan(cast_x):
            return False
        else:
            return True
    except:
        return False