

class Bcolors:
    COLORS = {
        "header": '\033[95m',
        "blue": '\033[94m',
        "cyan": '\033[96m',
        "green": '\033[92m',
        "warning": '\033[93m',
        "fail": '\033[91m',
        "end": '\033[0m',
        "bold": '\033[1m',
        "underline": '\033[4m'
    }
    

def wc(color, text):
    return f"{Bcolors.COLORS[color]}{text}{Bcolors.COLORS['end']}"