# Sentiment-Analysis
<div align="center">
  <img src="https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2021/06/sentimentanalysishotelgeneric-2048x803-1.jpg" alt="Analyse des Sentiments" width="800">
</div>

### Description

> ***Projet de Fin d'Études : Analyse de Sentiments et des Émotions avec les Données Twitter***

Ce repository contient tout le travail réalisé dans le cadre de mon projet de fin d'études pour l'obtention de ma Licence en Ingénierie des Données et Développement Logiciel. Le projet porte sur l'utilisation de techniques de traitement automatique du langage naturel (NLP) pour analyser les sentiments et les émotions exprimés dans les tweets.

#### Contenu 
| **Datasets**                                                                                                                                              | **Notebooks**                                                                                                         | **Presentation**                                   | **Report**                                            | **Models Comparison**                                                                                 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| -  le dataset utilisé pour l'analyse des sentiments. <br> - Le dataset d'émotions est très volumineux. Lien : [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/emotions).          | - Scripts Python utilisant TensorFlow, Keras, Pandas, NumPy, et Seaborn. <br> - Analyse des données et entraînement. | - Diapositives présentées lors de la soutenance. | - Rapport complet avec le code source LaTeX.        | - Comparaison et interprétation des modèles (Logistic Regression, Naive Bayes, RNN-LSTM, etc.).       |
 
#### Enoncé du problème
Avec l'augmentation exponentielle des utilisateurs des réseaux sociaux, des millions de tweets sont publiés chaque jour, contenant des opinions et des sentiments sur divers sujets. Ces données précieuses restent souvent inexploitées en raison de leur format non structuré. Les entreprises et chercheurs manquent d’outils efficaces pour extraire des insights utiles à partir de ces tweets. Ce projet vise à développer une solution basée sur des techniques de machine learning et NLP pour analyser automatiquement les sentiments et émotions exprimés dans les tweets, permettant ainsi de mieux comprendre les tendances et opinions publiques.

#### Etude Comparative des Modèles
Dans ce projet, nous avons effectué une étude comparative entre plusieurs modèles de Machine Learning (ML) et Deep Learning (DL). Les modèles étudiés comprenaient la Régression Logistique, Naive Bayes, RNN-LSTM, ainsi que d'autres techniques pertinentes. Chaque modèle a été évalué en termes de précision, de rappel, de F-mesure et d'autres métriques pertinentes. Nous avons analysé les performances de chaque modèle en fonction de leur capacité à prédire correctement les sentiments et les émotions dans les textes. À la fin de l'étude comparative, nous avons sélectionné le modèle ayant les meilleures performances, c'est-à-dire celui offrant les résultats les plus fiables et précis pour l'analyse des sentiments et des émotions. Ce modèle a ensuite été intégré dans notre code pour effectuer les prédictions finales.

#### NLP: Natural Language Processing (Traitement du Langage Naturel)

<div align="center">
  <img src="https://www.blumeglobal.com/media/wp-content/uploads/2018/11/NLP-image-scaled.jpg?rnd=133498791419900000" alt="Tokenisation" width="800">
</div>


Nous avons des techniques de Traitement du Langage Naturel (TLN) comme élément clé de l'analyse avec l'apprentissage automatique. Le traitement du langage naturel permet de convertir les mots présents dans le texte en vecteurs mathématiques, nécessaires au bon fonctionnement des modèles d'apprentissage automatique. Une fois ces vecteurs mathématiques générés, ils sont transmis aux modèles d'apprentissage automatique respectifs pour la prédiction. Ces caractéristiques, ainsi que certaines nouvelles fonctionnalités créées, seront utilisées par les modèles d'apprentissage automatique et d'apprentissage profond pour générer des prédictions précises. Cette approche garantit que les modèles reçoivent les informations les plus pertinentes du texte et fournissent les meilleurs résultats possibles.  

#### Prétraitement des Données
Le prétraitement des données est une étape essentielle dans tout projet d’analyse ou d’apprentissage automatique. Cette phase vise à transformer les données brutes en un format utilisable et cohérent, tout en éliminant les erreurs ou incohérences pouvant nuire à la performance des modèles.

Dans notre projet, on a suivi un processus structuré de prétraitement afin d'assurer que les données soient prêtes pour l'analyse et la modélisation. Voici les principales étapes effectuées :

> ça sera un résumé de toutes les étapes qu'on a suivi, trouver les explications en détails dans ce fichier [Rapport](https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/tree/main/Report)

### Nettoyage des Données
- **Vérification du type de données** : S'assurer que les colonnes ont les types appropriés (ex. : numérique, catégorique).
- **Gestion des valeurs manquantes** : Imputation des valeurs manquantes en utilisant la moyenne ou la médiane.
- **Détection et correction des valeurs aberrantes** : Identification et traitement des valeurs inhabituelles.
- **Suppression des doublons** : Élimination des enregistrements redondants pour éviter les biais.
> [Trouver le tableau des résultats avant et après le nettoyage ici](https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/tree/main/Report)


### Préparation des Textes
- **Tokenisation** : Division des textes en unités lexicales (tokens) pour une analyse efficace.
  >
 <div align="center">
  <img src="https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/Report/Source%20Code/figures/Capture%20d%E2%80%99%C3%A9cran%20(692).png" alt="Tokenisation" width="800">
</div>

- **Conversion** : Transformation des textes en minuscules (majiscules) pour éviter les erreurs liées à la casse.
>
<div align="center">
  <img src="https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/Report/Source%20Code/figures/Capture%20d%E2%80%99%C3%A9cran%20(693).png" alt="Tokenisation" width="800">
</div>

- **Suppression des mots vides** : Retrait des mots non significatifs comme *"le"*, *"et"*, etc., afin de réduire le bruit.
>
<div align="center">
  <img src="https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/Report/Source%20Code/figures/Capture%20d%E2%80%99%C3%A9cran%20(694).png" alt="Tokenisation" width="800">
</div>

- **Stemming et lemmatisation** : Simplification des mots en leur forme de base pour normaliser le texte.
Voici des représentations visuelles du Stemming et de la Lemmatisation, illustrant leurs effets respectifs sur le prétraitement des textes.
>
<p align="center">
  <img src="https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/Report/Source%20Code/figures/Stemming_53678d43bc.png" alt="Stemming" width="400">
  <img src="https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/Report/Source%20Code/figures/Lemmatization_5338fc7c3e.png" alt="Lemmatization" width="400">
</p>

Nous avons essayé les deux techniques  pour le traitement des mots dans le texte : Stemming et Lemmatisation, afin de comparer leur impact sur les performances des modèles. Cependant, dans notre cas, les résultats étaient assez similaires, et nous n'avons pas observé de différence notable en termes de précision des modèles entre les deux approches. Cela suggère que, pour notre jeu de données, les deux méthodes ont un effet comparable sur la qualité du modèle.
> [Trouver la comparaison des résultats dans ce fichier](https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/tree/main/models%20comparaisons)

### Division des données
Lors de la préparation des données, la méthode de division varie en fonction du type de modèle utilisé :

#### Machine Learning (ML)
- Les données sont généralement divisées en deux ensembles : **Entraînement (train)** et **Test (test)**.
- **Raison** : La plupart des modèles ML ne nécessitent pas un troisième ensemble, car les hyperparamètres peuvent être optimisés directement sur l'ensemble d'entraînement en utilisant des techniques comme la validation croisée.
  
       X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X, y, test_size=0.2, random_state=42)

#### Deep Learning (DL)
- Les données sont divisées en trois ensembles : **Entraînement (train)**, **Validation (val)**, et **Test (test)**.
- **Raison** :
  - Les modèles DL possèdent souvent un grand nombre de paramètres et hyperparamètres à ajuster (par exemple, la taille du réseau, le taux d'apprentissage, etc.).
  - L'ensemble de **validation** est utilisé pour évaluer les performances du modèle après chaque itération d'entraînement (epoch). Cela permet :
    - D'éviter le surapprentissage (*overfitting*) sur l'ensemble d'entraînement.
    - De sélectionner les meilleurs hyperparamètres sans biaiser l'évaluation finale.
  - L'ensemble de **test** est réservé à l'évaluation finale du modèle, garantissant une estimation impartiale de ses performances.

        X_train_dl, X_val_dl, X_test_dl, y_train_dl, y_val_dl, y_test_dl = train_test_split(train_test_split(X, y, test_size=0.3, random_state=42), test_size=0.5, random_state=42)

### Extraction des Caractéristiques
- Transformation des textes en vecteurs numériques à l’aide de techniques telles que :
  
  - **CountVectorizer**
  >
<div align="center">
  <img src="https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/Report/Source%20Code/figures/count%20vectorizer.png" alt="Tokenisation" width="800">
</div>

  - **TF-IDF (Term Frequency - Inverse Document Frequency)**

<div align="center">
  <img src="https://enjoymachinelearning.com/wp-content/uploads/2022/10/tfidfvscount_image1.png" alt="Tokenisation" width="800">
</div>

Nous avons essayé les deux méthodes pour la vectorisation des textes : TF-IDF (Term Frequency-Inverse Document Frequency) et CountVectorizer (compteur de mots), afin de comparer leurs performances respectives sur nos modèles. Cependant, dans notre cas, nous n'avons pas observé de différence significative entre les deux approches. Les résultats étaient relativement similaires, ce qui suggère que, pour notre jeu de données spécifique, les deux méthodes produisent des performances équivalentes en termes de précision des modèles.
> [Trouver les resultats dans ce fichier](https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/tree/main/models%20comparaisons)




     
 




    

     

     



