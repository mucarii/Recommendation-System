# %%[importando bibliotecas]
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# %%[carregando os dados]
data = fetch_movielens(min_rating=4.0)

# %%[printando e testando]
print(repr(data["train"]))
print(repr(data["test"]))

# %%[criando o modelo]
model = LightFM(loss="warp")

# %%[treinando o modelo]
model.fit(data["train"], epochs=30, num_threads=2)


# %%[recomendando os usuários]
def sample_recommendation(model, data, user_ids):
    # numero de usuarios e filmes
    n_users, n_items = data["train"].shape

    # gerando as recomendações para cada usuário
    for user_id in user_ids:
        # filmes que o usuário já gostou
        known_positives = data["item_labels"][data["train"].tocsr()[user_id].indices]
        # recomendações para o usuário
        scores = model.predict(user_id, np.arange(n_items))
        # ranking dos recomendados
        top_items = data["item_labels"][np.argsort(-scores)]

        # printando as recomendações para o usuário
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)
        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


# %%
