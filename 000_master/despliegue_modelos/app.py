import asyncio
import csv
import time
import os
from datetime import datetime
from configparser import ConfigParser
from random import randint
import threading

import torch
import pandas as pd
import matplotlib.pyplot as plt

# Librería para análisis de sentimientos
from transformers import pipeline

from twikit import Client, TooManyRequests

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


# Usar la extension edit cookies y luego el script 00_cookies.py
CSV_FILENAME = 'tweets_noboa.csv'
COOKIES_FILENAME = 'cookies.json'

# =======================================
# CONFIG: LEE NUESTRAS CREDENCIALES
# =======================================
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


async def recolectar_tweets():
    """
    - Carga/inicia sesión en Twitter usando twikit.
    - Carga un set con IDs de tuits existentes (para no duplicar).
    - Itera sobre cada keyword y busca tuits.
    - Guarda tuits en un CSV, evitando duplicados.
    - Repite periódicamente para simular 'tiempo real'.
    """
    client = Client(language='es-MX')
    
    # Verificar si ya hay cookies guardadas
    try:
        client.load_cookies(COOKIES_FILENAME)
        print("Cookies cargadas correctamente.")
    except FileNotFoundError:
        print("No se encontraron cookies. Iniciando sesión...")
        await client.login(auth_info_1=username, auth_info_2=email, password=password)
        client.save_cookies(COOKIES_FILENAME)
        print(f"Cookies guardadas en '{COOKIES_FILENAME}'.")

    while True:
        try:
            # Cargar IDs de tuits existentes para evitar duplicados
            existing_tweet_ids = set()
            file_exists = os.path.isfile(CSV_FILENAME)
            if file_exists:
                with open(CSV_FILENAME, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_tweet_ids.add(row['tweet_id'])

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
                                    product='Latest'  # 'Top' o 'Latest'
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
                            seconds_to_wait = max(wait_time.total_seconds(), 0)
                            time.sleep(seconds_to_wait)
                            continue
                    
                    print(f"Se guardaron {total_encontrados} tuits nuevos para '{keyword}'.")

            print("Recolección completa de esta ronda. Esperando un rato para volver a recolectar...\n")
            
            # Esperar X tiempo antes de volver a buscar (simulando 'tiempo real')
            time.sleep(60)  # cada minuto, por ejemplo

        except Exception as e:
            print(f"Error en la recolección: {e}")
            # Esperamos un tiempo y volvemos a intentarlo
            time.sleep(30)


device = 0 if torch.cuda.is_available() else -1
print(f"Device (sentiment model): {device}")
modelo_sentimiento = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment", 
    device=device
)

def analizar_sentimiento_avanzado(texto):
    try:
        resultado = modelo_sentimiento(texto[:512])
        # La etiqueta es '1 star', '2 stars', '3 stars', '4 stars', '5 stars'
        etiqueta = int(resultado[0]['label'].lower().split()[0])
        if etiqueta > 3:
            return "positivo"
        elif etiqueta < 3:
            return "negativo"
        else:
            return "neutral"
    except Exception as e:
        print(f"Error al procesar el texto: {texto} - {e}")
        return "error"

# LEER CSV, ANALIZAR Y GRAFICAR
def loop_analisis_y_graficacion():
    """
    En un bucle infinito:
      1) Lee el CSV (tweets_noboa.csv).
      2) Aplica el análisis de sentimiento a los tweets que no lo tengan (o reanaliza todo para simplificar).
      3) Cuenta los resultados y muestra una gráfica de barras con plt.
      4) Espera unos segundos, y repite.
    """
    plt.ion()  # modo interactivo ON para refrescar la misma ventana
    fig, ax = plt.subplots(figsize=(8,6))

    while True:
        try:
            if not os.path.isfile(CSV_FILENAME):
                print(f"No se encuentra '{CSV_FILENAME}'. Esperando que el productor genere datos...")
                time.sleep(10)
                continue

            df = pd.read_csv(CSV_FILENAME)

            # Si ya existe una columna de "sentimiento", la usamos. Sino, la creamos.
            df["sentimiento"] = df["text"].apply(analizar_sentimiento_avanzado)

            # Contar los tweets por sentimiento
            conteo_sentimientos = df["sentimiento"].value_counts()
            positivos = conteo_sentimientos.get("positivo", 0)
            negativos = conteo_sentimientos.get("negativo", 0)
            neutrales = conteo_sentimientos.get("neutral", 0)

            # Cálculo del índice de popularidad
            total = positivos + negativos + neutrales
            popularidad = (positivos - negativos) / total if total > 0 else 0

            print("=== Análisis de Sentimientos ===")
            print(f"Positivos: {positivos}, Negativos: {negativos}, Neutrales: {neutrales}")
            print(f"Índice de Popularidad: {popularidad:.2f}\n")

            # Graficar
            ax.clear()
            ax.bar(
                ["Positivo", "Negativo", "Neutral"], 
                [positivos, negativos, neutrales],
                color=["green", "red", "gray"]
            )
            ax.set_xlabel("Sentimiento")
            ax.set_ylabel("Cantidad de Tweets")
            ax.set_title("Distribución de Sentimientos en Twitter sobre el Presidente de Ecuador")
            ax.set_ylim(0, max(positivos, negativos, neutrales) + 5 if total > 0 else 5)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            plt.draw()
            plt.pause(5)  # pausar 5 segundos; en este tiempo la ventana se refresca

        except Exception as e:
            print(f"Error en análisis/graficación: {e}")
            time.sleep(5)


# =======================================
def main():
    # 1) Thread para recolección (producer)
    #    Arrancará la corrutina que hace asyncio.run(recolectar_tweets())
    def run_producer():
        asyncio.run(recolectar_tweets())
    
    producer_thread = threading.Thread(target=run_producer, daemon=True)
    producer_thread.start()

    # 2) Bucle principal de análisis y graficación (consumer)
    loop_analisis_y_graficacion()

if __name__ == "__main__":
    main()
