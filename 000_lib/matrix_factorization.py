import math
import pandas as pd
import numpy as np


def generateModel(df_ratings:pd.DataFrame, embedding_dim=30, init_stddev=1, num_iterations=500, learning_rate=0.03, verbosity=1):
    """
    df_ratings: have columns ["user","item","rating"]
    """
    column_user = "user"
    column_item = "item"
    column_rating = "rating"

    last_loss_result = math.inf

    try:

        users_count = int(np.max(df_ratings['id'])) + 1  # se le redujo 1
        items_count = int(np.max(df_ratings['id_product'])) + 1
        log("ratings['id_product']", ratings['id_product'].values)
        log("ratings['id_product']", len(ratings['id_product'].values))
        log("ratings['id_product']", np.max(ratings['id_product'].values))
        log("ITEMS COUNT", items_count)

        # Build the CF model and train it.
        Matrix_A_test, model = build_model(ratings, users_count, items_count, embedding_dim=embedding_dim, init_stddev=init_stddev, verbosity=verbosity)  # embedding_dim=30  num_iterations=1000
        U, V = model.train(num_iterations=num_iterations, learning_rate=learning_rate, verbosity=verbosity)

        # El error se mide con una matriz de test que no ha sido mostrada al modelo
        cambiarMatrizReferencia(Matrix_A_test)

        model.minimum_test_loss = sparse_mean_square_error(U, V)
        if verbosity == 1:
            log("test_loss final obtenido", model.minimum_test_loss)

        save_model(model, ratings)

        """
        if last_loss_result == math.inf:
            save_model(model, ratings)
        else:
            if model.minimum_test_loss < last_loss_result:
                # print("El modelo mejoró por un ", ((last_loss_result-model.minimum_test_loss)*100)/last_loss_result, " % respecto al minimo inicial")
                save_model(model, ratings)
            # else:
                # print("El modelo empeoró un ", ((model.minimum_test_loss-last_loss_result)*100)/last_loss_result, " % respecto al minimo inicial")
        """
    except Exception as e:
        raise e