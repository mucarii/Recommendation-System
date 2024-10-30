# Sistema de Recomendação de Filmes com LightFM

Este projeto utiliza o algoritmo LightFM para construir um sistema de recomendação de filmes com base em classificações de usuários. Ele é capaz de recomendar filmes que os usuários ainda não assistiram, mas que têm alta probabilidade de gostar.

## Funcionalidades

- Carrega um conjunto de dados de filmes e classificações de usuários.
- Treina um modelo de recomendação utilizando a perda "warp" (Weighted Approximate-Rank Pairwise).
- Gera recomendações personalizadas para usuários com base em suas classificações.

## Tecnologias Utilizadas

- Python 3.11
- NumPy
- LightFM
- Scikit-learn (para manipulação de dados)
