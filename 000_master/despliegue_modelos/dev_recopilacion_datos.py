import asyncio
import csv
import time
import os
from datetime import datetime
from configparser import ConfigParser
from random import randint

from twikit import Client, TooManyRequests

# ===================================
# CONFIGURACIÓN Y LISTA DE KEYWORDS
# ===================================
keywords = [
    "Daniel Noboa",
    "Presidente Daniel Noboa",
    "Presidente del Ecuador",
    "@DanielNoboaOk",
    "@Presidencia_Ec",
    "#DanielNoboa",
    "#PresidenteNoboa",
    "#PresidenteEcuador",
    "#GobiernoEcuador",
    "#ElNuevoEcuadorResuelve",
    "Inseguridad en Ecuador",
    "Crisis energética en Ecuador",
    "Política ecuatoriana",
    "Elecciones Ecuador 2025",
    "#Ecuador",
    "#PolíticaEcuador",
    "#NoticiasEcuador"
]

CSV_FILENAME = 'tweets_noboa.csv'

# ===================================
# LECTURA DE CREDENCIALES
# ===================================
config = ConfigParser()
config.read('config.ini')

if 'X' in config:
    username = config['X'].get('username')
    email = config['X'].get('email')
    password = config['X'].get('password')
else:
    raise ValueError(
        "Archivo config.ini mal configurado. "
        "Asegúrate de incluir la sección [X] con username, email y password."
    )

# ===================================
# FUNCIÓN PRINCIPAL
# ===================================
async def main():
    """
    - Carga/inicia sesión en Twitter usando twikit.
    - Carga un set con IDs de tuits existentes (para no duplicar).
    - Itera sobre cada keyword y busca tuits.
    - Guarda tuits en un CSV, evitando duplicados.
    """
    # Instanciar el cliente
    client = Client(language='es-MX')
    
    # Verificar si ya hay cookies guardadas
    try:
        client.load_cookies('cookies.json')
        print("Cookies cargadas correctamente.")
    except FileNotFoundError:
        print("No se encontraron cookies. Iniciando sesión...")
        await client.login(auth_info_1=username, auth_info_2=email, password=password)
        client.save_cookies('cookies.json')
        print("Cookies guardadas en 'cookies.json'.")

    # Cargar IDs de tuits existentes para evitar duplicados
    existing_tweet_ids = set()
    file_exists = os.path.isfile(CSV_FILENAME)
    if file_exists:
        with open(CSV_FILENAME, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_tweet_ids.add(row['tweet_id'])  # usaremos 'tweet_id' como nombre de la columna

    # Abrir el CSV en modo append (para escribir nuevos tuits)
    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Si el archivo no existía, escribir la cabecera
        if not file_exists:
            writer.writerow([
                "tweet_id",
                "username",
                "user_id",
                "text",
                "created_at",
                "retweet_count",
                "favorite_count"
            ])
        
        # Iterar sobre cada palabra clave
        for keyword in keywords:
            print(f"\n=== Obteniendo tuits para: '{keyword}' ===")
            
            # Usamos un objeto para la paginación (search_tweet retorna un 'TweetPaginator')
            tweets_paginator = None
            total_encontrados = 0

            while True:
                try:
                    # Primera búsqueda o siguiente página
                    if tweets_paginator is None:
                        tweets_paginator = await client.search_tweet(
                            query=keyword,
                            product='Latest'  # Puedes usar 'Top' o 'Latest'
                        )
                    else:
                        # Retardo para simular acción humana y evitar bloqueos
                        wait_time = randint(3, 8)
                        print(f"Esperando {wait_time} segundos antes de la siguiente página...")
                        time.sleep(wait_time)
                        
                        tweets_paginator = await tweets_paginator.next()
                    
                    # Si no hay más tuits, break
                    if not tweets_paginator:
                        print(f"Ya no hay más tuits para '{keyword}'.")
                        break
                    
                    # Recorrer los tuits devueltos en esta página
                    for tweet in tweets_paginator:
                        # Verificar si ya lo teníamos guardado
                        if tweet.id not in existing_tweet_ids:
                            # Guardar en CSV
                            writer.writerow([
                                tweet.id,
                                tweet.user.name,
                                tweet.user.id,
                                tweet.text.replace('\n', ' ').strip(),
                                tweet.created_at,
                                tweet.retweet_count,
                                tweet.favorite_count
                            ])
                            # Agregar a la memoria de duplicados
                            existing_tweet_ids.add(tweet.id)
                            total_encontrados += 1
                
                except TooManyRequests as e:
                    # Manejo de límite de peticiones
                    rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                    print(f"Se alcanzó el límite de solicitudes. Esperando hasta {rate_limit_reset}...")
                    wait_time = rate_limit_reset - datetime.now()
                    # Asegurarnos de no pasar un valor negativo en caso de desajuste horario
                    seconds_to_wait = max(wait_time.total_seconds(), 0)
                    time.sleep(seconds_to_wait)
                    continue  # Reintentar tras la espera
                
                # Puedes establecer aquí un criterio de parada,
                # por ejemplo, si ya alcanzaste N tuits para esta keyword,
                # o si quieres limitar las páginas. De lo contrario, 
                # el while seguirá hasta que no haya más páginas.
            
            print(f"Se guardaron {total_encontrados} tuits nuevos para la keyword '{keyword}'.")

    print("Finalizado: la recolección de tuits ha concluido.")

# ===================================
# EJECUCIÓN
# ===================================
if __name__ == "__main__":
    asyncio.run(main())
