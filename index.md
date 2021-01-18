
 Diplômée d'un master en Data Science, option santé de l'université Paris-Saclay. Je présenterai dans ce site plusieurs article traitant des statistiques, de Machine Learning, Deep Learning.


# [Analyse de sentiments avec PyTorch — Code complet](art1.md)
## Attention et BiLSTM pour le NLP

Dans cet article, je vais présenter une tâche d’analyse de sentiments sur le jeu de données IMDb Reviews, en utilisant un modèle d’attention et de BiLSTM, codé avec PyTorch.

### Les données IMDb Reviews

On utilise ici une base de données bien connue provenant de IMDb Reviews téléchargeable en suivant le lien [ici](http://ai.stanford.edu/~amaas/data/sentiment/).

Cette base de données est composée de 50 000 commentaires de films labellisés 0 si le commentaire est négatif et 1 s’il est positif.

Ci-dessous, on donne un exemple de commentaire positif.

    This movie will always be a Broadway and Movie classic, as long as there are still people who sing, dance, and act.

Et un exemple de commentaire négatif :

    no comment — stupid movie, acting average or worse… screenplay — no sense at all… SKIP IT!

La base de données a été préparée au préalable et se compose de 25 000 commentaires dans l’échantillon d’entraînement et 25 000 commentaires dans l’échantillon de test. Par ailleurs, les données sont équilibrées car elles sont composées de 50% de commentaires positifs et 50% de commentaires négatifs.

![alt text](https://miro.medium.com/max/425/1*eNGHL7956O3YuJayHp7dJQ.png)

### Import des données IMDb avec PyTorch

Pour importer ces données, c’est très simple : il suffit d’implémenter le code suivant :

![alt text](https://miro.medium.com/max/700/1*kfbl90T6R2SQFZ5VJEBbgA.png)

![alt text](https://miro.medium.com/max/700/1*J-JxFPQ0pi6aLX4sRD-hig.png)

TorchText a une méthode Field qui sert à définir comment les données brutes doivent être traitées.

La méthode TEXT définit comment les commentaires doivent être traités, et LABEL comment les labels doivent être traités. Ces méthodes comportent plusieurs paramètres qui sont décrits dans la [documentation](https://torchtext.readthedocs.io/en/latest/data.html#field).

![alt text](https://miro.medium.com/max/700/1*NeVMBqsyeKcomTUKg1QiUQ.png)

Les données sont importées et on crée un échantillon de validation avec la méthode .split .

### Word Embedding : GloVe

Pour pouvoir représenter nos données textuelles en représentation numérique, nous utilisons une méthode de plongement (ou embedding) dans un espace de dimension finie, typiquement autour de 50 à 300 dimensions. Ces méthodes de plongement permettent d’obtenir une représentation des mots dans un espace avec une forme de similarité entre eux, dans lesquelles le sens des mots les rapproche dans cet espace, en terme de distances statistiques.

Pour notre étude, nous avons utilisé la méthode GloVe qui est un algorithme d’apprentissage non supervisé qui permet d’obtenir des représentations vectorielles des mots. L’entraînement a été réalisé sur un corpus de texte de 6 milliards de mots, provenant de Wikipédia.

L’hypothèse principale de cette technique étant de prendre en compte le contexte dans lequel le mot a été trouvé, c’est-à-dire les mots avec lesquels il est souvent utilisé.
On peut ainsi retrouver beaucoup de régularités linguistiques simplement en effectuant des translation linéaires dans cet espace de représentation, comme illustré ci-dessous.

![alt text](https://miro.medium.com/max/700/1*j_FJbKJuRS7-Iu7p720z7g.jpeg)

On observe dans la figure ci-dessous que les mots sister, uncle, brother appartiennent au lexique de la famille et sont proches dans l’espace vectoriel représenté. De plus, les différences de vecteur tels que man — woman, king — queen et encore brother — sister sont à peu près égales.

### Avec PyTorch

![alt text](https://miro.medium.com/max/700/1*r3xFNyYuHQxBFo-RUUx_Ow.png)

On construit le vocabulaire issu de l’échantillon train avec la méthode build_vocab. On utilise la représentation GloVe de dimension 100.

![alt text](https://miro.medium.com/max/700/1*nWohBrCrRkp_2TmnGNp-Bg.png)

La dernière étape de la préparation des données consiste à créer les itérateurs. Nous les parcourons dans la boucle d’apprentissage / d’évaluation, et ils retournent ici un batch (ou lot) de taille 32, à chaque itération.

On utilise BucketIterator qui est un itérateur qui renverra un batch de séquences de mêmes longueurs, en prenant la longueur de la séquence la plus longue du batch et en ajoutant des zéros aux séquences plus courtes afin qu’elles aient toutes la même longueur. On appelle cette méthode d’ajout de zéros : le padding.

### BiLSTM

Nous utilisons ici un réseau de neurones récurrents bidirectionnel (ou BiLSTM). Ce type de réseau de neurones permet de prendre en compte le contexte d’une phrase en considérant que les mots d’un commentaires ne sont pas indépendants entre eux.

Un réseau de neurones récurrent est typiquement organisé en couches successives dont les cellules (ou neurones) sont connectés aux couches suivantes et contiennent des valeurs variant en fonction du temps. Concrètement, le modèle BiLSTM prend une séquence de mots, un à la fois, et produit un état caché, pour chaque mot. On utilise ce modèle de manière itérative en lui donnant le mot courant ainsi que l’état caché du mot précédent, pour produire l’état caché suivant. On répète le processus pour tous les mots d’un commentaire. En plus d’avoir un modèle BiLSTM traitant les mots du commentaire du premier au dernier (BiLSTM forward), on introduit un second modèle BiLSTM traitant les mots du dernier au premier (BiLSTM backward). La prédiction du sentiment est réalisée en utilisant une concaténation du dernier état caché du BiLSTM forward (obtenu à partir du dernier mot de la phrase), et le dernier état caché du BiLSTM backward (obtenu à partir du premier mot de la phrase), on le donne à une couche fully connected (en gris sur l’image ci-dessous), pour recevoir notre sentiment prédit.

![alt text](https://miro.medium.com/max/438/1*NxYCTTx5MxRU07GVc-3GGw.png)

### Attention

Les mécanismes d’attention, et plus particulièrement les mécanismes d’auto-attention, permettent de déterminer l’importance des mots de la séquence d’entrée pour la classification en considérant la similarité entre ces mots. Ces mécanismes ont l’avantage de pouvoir faire des liens entre des éléments très distants d’une séquence. Étant donné que tous les mots d’un commentaire ne contribuent pas de manière égale à la représentation du commentaire, le mécanisme d’auto-attention est utilisé pour extraire les mots importants en leur donnant un poids plus élevé.

### Le modèle BiLSTM+Attention avec PyTorch

Le modèle est composé :
 - d’une couche embedding GloVe
 - d’une couche BiLSTM
 - d’une couche Attention
 - de deux couches fully connected
 
 ![alt text](https://miro.medium.com/max/622/1*I1jaKZAdODIkpE6ZZIa6zw.png)
 
 ![alt text](https://miro.medium.com/max/700/1*68YJs94kP55X5Mkb4125_w.png)
 
 ![alt text](https://miro.medium.com/max/700/1*uj71Mac5BDRmH9wyJrD-zw.png)
 
 On choisit ensuite les hyperparamètres suivants : 
 
 ![alt text](https://miro.medium.com/max/700/1*NUrekZWkcDakomCCJVUAbg.png)
 
 On peut calculer le nombre de paramètres à entraîner :
 
 ![alt text](https://miro.medium.com/max/627/1*ctKRc5hGiWNlbrfqADmkSA.png)
  
 On utilise l’optimiseur Adam. La fonction de perte utilisée est ici la binary cross entropy with logits.

En utilisant .to, nous pouvons placer le modèle et le critère sur le GPU (si nous en avons un).

La fonction criterion calcule la perte, cependant nous devons écrire notre fonction pour calculer l’accuracy qui nous permettra de savoir si notre modèle est performant.

![alt text](https://miro.medium.com/max/700/1*vmQF_Z-LqNMc5rfoWrch8w.png)

La fonction train itère sur tous les exemples, un batch à la fois. model.train()est utilisé pour mettre le modèle en "mode entraînement", ce qui active dropout.

Pour chaque batch, on met à zéro le gradient car PyTorch ne met pas à zéro les gradients automatiquement. Chaque paramètre dans un modèle a un attribut grad qui stocke le gradient calculé par criterion. Nous introduisons ensuite le batch de phrases, batch.text, dans le modèle.

La perte et la précision sont ensuite calculées à l'aide de nos prédictions et des labels, batch.label, la perte étant moyennée sur toutes les séquences du batch.

Nous calculons le gradient de chaque paramètre avec loss.backward (), puis mettons à jour les paramètres en utilisant les gradients et l'algorithme d'optimisation avec optimizer.step ().

Enfin, nous retournons la perte et l’accuracy moyennées sur toute l’époque.

![alt text](https://miro.medium.com/max/677/1*tp-UrfLyea5vXiC4PL1PbA.png)

La fonction evaluate est similaire àtrain, avec quelques modifications car on ne veut pas mettre à jour les paramètres lors de l'évaluation.

model.eval () met le modèle en "mode d'évaluation", ceci désactive dropout.

Aucun gradient n’est calculé sur les opérations PyTorch à l’intérieur du bloc with no_grad (). Cela réduit l'utilisation de la mémoire et accélère le calcul.

Le reste de la fonction est identique à train, avec la suppression deoptimizer.zero_grad (),loss.backward ()etoptimizer.step (), car nous ne mettons pas à jour les paramètres du modèle lors de l'évaluation.

![alt text](https://miro.medium.com/max/626/1*HhBnCMNbIrMUSEKc2NpfhQ.png)

Nous entraînons ensuite le modèle sur plusieurs époques, une époque étant un passage complet à travers toutes les séquences dans les ensembles d’apprentissage et de validation.

À chaque époque, si la perte de validation est la meilleure que nous ayons vue jusqu’à présent, nous enregistrerons les paramètres du modèle, puis une fois l’entraînement terminé, nous utiliserons ce modèle sur l’ensemble de test.

### Entraînement

![alt text](https://miro.medium.com/max/700/1*qeiv-IdSPStd16p_fayFDQ.png)

![alt text](https://miro.medium.com/max/412/1*5w8W0xN-2lgGrahhrhyiBA.png)

![alt text](https://miro.medium.com/max/356/1*S3qvund0IUGYRCpkXNAaQQ.png)

![alt text](https://miro.medium.com/max/397/1*kM0l5sTdTSB065uNS61bKw.png)

![alt text](https://miro.medium.com/max/390/1*IVeJldeZALhfb27PVffUGA.png)

![alt text](https://miro.medium.com/max/396/1*ZoyHdB2BTcxeRgUv-wVkIA.png)

En 10 époques, on voit que la loss a bien diminué et que l’accuracy atteint environ 83 % sur l’échantillon train et validation.

### Sur l’échantillon test

![alt text](https://miro.medium.com/max/654/1*MtfS15-TZUBILWYHVm5yLg.png)

### Inférence

Pour pouvoir tester notre modèle avec des commentaires de notre choix on écrit une fonction predict_sentiment qui :

    définit le modèle en mode d’évaluation
    tokenise la phrase, c’est-à-dire la divise d’une chaîne brute en une liste de tokens
    indexe les tokens en les convertissant en leur représentation numérique à partir de notre vocabulaire
    obtient la longueur de notre séquence
    convertit les index, qui sont une liste Python en un tenseur PyTorch
    convertit la longueur en un tenseur
    obtient la prédiction de sortie d’un nombre réel compris entre 0 et 1
    convertit le tenseur contenant une valeur unique en un entier avec la méthode item ()

Nous nous attendons à ce que les avis avec un sentiment négatif renvoient une valeur proche de 0 et les avis positifs renvoient une valeur proche de 1.

![alt text](https://miro.medium.com/max/602/1*oP1g_HaZJvm2Tr_IleAARg.png)

On remarque que notre modèle arrive à prédire le sentiment de ces commentaires, et il reconnaît la négation.

Le notebook complet peut être trouvé [ici](https://github.com/aminaghoul/sentiment-analysis/blob/master/5b_AttentionLSTM.ipynb).

### Pour la suite :

Plusieurs autres modèles ont été implémentés pour cette tâche d’analyse de sentiments comme :

 - [Machine Learning (SVM, Régression Logistique) avec sklearn](https://github.com/aminaghoul/sentiment-analysis/blob/master/0-MachineLearning.ipynb)
 - [RNN simple](https://github.com/aminaghoul/sentiment-analysis/blob/master/1_RNN_Simple.ipynb)
 - [BiLSTM](https://github.com/aminaghoul/sentiment-analysis/blob/master/2a_LSTM.ipynb)
 - [GRU](https://github.com/aminaghoul/sentiment-analysis/blob/master/2b_GRU.ipynb)
 - [FastText](https://github.com/aminaghoul/sentiment-analysis/blob/master/3-FastText.ipynb)
 - [CNN](https://github.com/aminaghoul/sentiment-analysis/blob/master/4-CNN.ipynb)
 - [Attention + GRU](https://github.com/aminaghoul/sentiment-analysis/blob/master/5a_AttentionGRU.ipynb)
 - [Transformers + BiLSTM](https://github.com/aminaghoul/sentiment-analysis/blob/master/6b-TransformersLSTM.ipynb)
 - [Transformers + GRU](https://github.com/aminaghoul/sentiment-analysis/blob/master/6a-TransformersGRU.ipynb)
 - [Transformers (from scratch)](https://github.com/aminaghoul/sentiment-analysis/blob/master/7-Transorfmers.ipynb)
 - [BERT](https://github.com/aminaghoul/sentiment-analysis/blob/master/8-BERT.ipynb)
 
 ### Références :

    https://github.com/aminaghoul/sentiment-analysis
    https://github.com/aminaghoul/sentiment-analysis/blob/master/5b_AttentionLSTM.ipynb
    https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
    https://www.aclweb.org/anthology/W18-6226/
