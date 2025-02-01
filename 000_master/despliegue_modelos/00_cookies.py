import json

# Ruta del archivo JSON exportado desde EditThisCookie
INPUT_FILE = 'cookies_exported.json'  # Reemplaza con el nombre de tu archivo
OUTPUT_FILE = 'cookies.json'

def convert_cookies(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        cookies = {}
        for cookie in data:
            name = cookie.get("name")
            value = cookie.get("value")
            if name and value:
                cookies[name] = value

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(cookies, file, indent=4)

        print(f"Cookies convertidas y guardadas en {output_file}")
    except Exception as e:
        print(f"Error al convertir las cookies: {e}")

# Convertir cookies
convert_cookies(INPUT_FILE, OUTPUT_FILE)