import tweepy
import os
from dotenv import load_dotenv
import pandas as pd

# Cargar credenciales desde .env
load_dotenv(".env")

# Credenciales de la API de Twitter
api_key = os.getenv("TWITTER_API_KEY")
api_secret_key = os.getenv("TWITTER_API_SECRET_KEY")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

# Configurar autenticación
auth = tweepy.OAuth1UserHandler(api_key, api_secret_key, access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

def leer_tweets(cuenta=None, consulta=None, limite=100):
    """
    Lee hasta 100 tweets usando la API de Twitter.
    - cuenta: str (opcional) -> Usuario de Twitter del que leer tweets (sin el @).
    - consulta: str (opcional) -> Consulta para buscar tweets (ej. palabras clave).
    - limite: int -> Máximo de tweets a leer (máximo permitido: 100).
    """
    tweets = []
    try:
        if cuenta:
            # Leer tweets del timeline de un usuario específico
            for tweet in tweepy.Cursor(api.user_timeline, screen_name=cuenta, tweet_mode="extended").items(limite):
                tweets.append([tweet.created_at, tweet.user.screen_name, tweet.full_text])
        elif consulta:
            # Buscar tweets basados en una consulta
            for tweet in tweepy.Cursor(api.search_tweets, q=consulta, lang="es", tweet_mode="extended").items(limite):
                tweets.append([tweet.created_at, tweet.user.screen_name, tweet.full_text])
        else:
            print("Debes proporcionar un nombre de cuenta o una consulta.")
            return None

        # Guardar resultados en un DataFrame
        df = pd.DataFrame(tweets, columns=["Fecha", "Usuario", "Tweet"])
        print(df.head())
        return df
    except tweepy.errors.TweepyException as e:
        print(f"Error al leer tweets: {e}")
        return None

if __name__ == "__main__":
    # Configura una cuenta o una consulta para probar
    cuenta = "DanielNoboaOk"  # Cambia por el usuario que desees
    consulta = "Presidente Ecuador -filter:retweets"  # Búsqueda con palabras clave

    # Leer tweets del timeline de una cuenta (máx. 100)
    df_tweets = leer_tweets(cuenta=cuenta, limite=10)  # Cambia `limite` si necesitas menos
    if df_tweets is not None:
        df_tweets.to_csv("tweets_cuenta.csv", index=False, encoding="utf-8")
        print("Tweets guardados en 'tweets_cuenta.csv'.")

    # Leer tweets de una búsqueda (máx. 100)
    df_busqueda = leer_tweets(consulta=consulta, limite=10)  # Cambia `limite` si necesitas menos
    if df_busqueda is not None:
        df_busqueda.to_csv("tweets_busqueda.csv", index=False, encoding="utf-8")
        print("Tweets guardados en 'tweets_busqueda.csv'.")
