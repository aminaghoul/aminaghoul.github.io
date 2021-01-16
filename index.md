
 Diplômée d'un master en Data Science, option santé de l'université Paris-Saclay. Je présenterai dans ce site plusieurs article traitant des statistiques, de Machine Learning, Deep Learning.


# Analyse de sentiments avec PyTorch — Code complet
## Attention et BiLSTM pour le NLP

Dans cet article, je vais présenter une tâche d’analyse de sentiments sur le jeu de données IMDb Reviews, en utilisant un modèle d’attention et de BiLSTM, codé avec PyTorch.
Les données IMDb Reviews

On utilise ici une base de données bien connue provenant de IMDb Reviews téléchargeable en suivant le lien ci-dessous.

http://ai.stanford.edu/~amaas/data/sentiment/

Cette base de données est composée de 50 000 commentaires de films labellisés 0 si le commentaire est négatif et 1 s’il est positif.

Ci-dessous, on donne un exemple de commentaire positif.

    This movie will always be a Broadway and Movie classic, as long as there are still people who sing, dance, and act.

Et un exemple de commentaire négatif :

    no comment — stupid movie, acting average or worse… screenplay — no sense at all… SKIP IT!

La base de données a été préparée au préalable et se compose de 25 000 commentaires dans l’échantillon d’entraînement et 25 000 commentaires dans l’échantillon de test. Par ailleurs, les données sont équilibrées car elles sont composées de 50% de commentaires positifs et 50% de commentaires négatifs.

