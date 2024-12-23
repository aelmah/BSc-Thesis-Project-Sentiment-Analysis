# Sentiment-Analysis & Emotion Detection
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
#### Description des Corpus
* ***Sentiments140-MV***:Le dataset des sentiments également connu sous le nom Sentiment140-MV, est un ensemble de données largement utilisé dans le domaine de l’analyse des sentiments. Il comprend des tweets collectés à partir de Twitter, annotés avec l’étiquette du sentiment correspondant, généralement positif et négatif. Chaque entrée données a les caractéristiques suivantes :
 
— **Ids :** un nombre entier qui représente l’id du tweet ;

— **Date :** le timestamp du tweet ;

— **Flag :** la requête qui a été utilisée pour récupérer le Tweet avec Twitter API. S’il n’y a pas de requête, cette valeur est NO QUERY.

— **Utilisateur :** le nom de l’utilisateur qui a publié le message ;

— **texte :** le contenu réel du tweet.
<div align="center">
  <img src="https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/Report/Source%20Code/figures/WhatsApp%20Image%202024-06-07%20%C3%A0%2020.42.44_c32b6e6d.jpg" alt="Tokenisation" width="800">
</div>

* ***Emotions***:  Le dataset Emotions est une collection de messages en anglais provenant de Twitter,minutieusement annotés avec six émotions fondamentales : la colère, la peur, la joie, l’amour, la tristesse et la surprise. Ce dataset constitue une ressource précieuse pour comprendre et analyser le spectre diversifié des émotions exprimées dans les textes courts sur les réseaux sociaux. Chaque entrée dans ce dataset se compose d’un segment de texte représentant un message Twitter et d’une étiquette correspondante indiquant l’émotion prédominante transmise. Les émotions sont classées en six catégories : la tristesse (0), la joie (1), l’amour (2), la colère (3), la peur (4) et la surprise (5).

<div align="center">
  <img src="https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/Report/Source%20Code/figures/repartition-emotion.png" alt="Tokenisation" width="800">
</div>
  

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

### Approches basées sur le Machine Learning et Deep Learning

Dans notre projet, nous avons utilisé une combinaison de modèles de Machine Learning (ML) et de Deep Learning (DL) pour l'analyse de sentiment. Voici un tableau récapitulatif des modèles testés :

| **Approche**         | **Modèles**                                  |
|----------------------|----------------------------------------------|
| **Machine Learning**  | Régression Logistique, Naive Bayes (Multinomial, Complémentaire), Adaboost, Nu-SVC |
| **Deep Learning**     | CNN (Convolutional Neural Network), RNN (Recurrent Neural Network), LSTM (Long Short-Term Memory) |

> [Vous trouvez la description de chaque modèle et sa méthode de fonctionnement ici](https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/tree/main/Report)

### Résultats

<div style="display: flex; justify-content: space-between; gap: 40px;">

  <!-- Tableau 1 : Analyse des Sentiments -->
  <div>
    <h3><strong>Analyse des Sentiments</strong></h3>
    <table border="1" cellpadding="5" cellspacing="0" style="width: 48%; display: inline-block;">
      <tr>
        <th>Classificateur</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th>
      </tr>
      <tr><td><strong>Model 1</strong>: RNN_LSTM</td><td>85%</td><td>85%</td><td>85%</td><td>85%</td></tr>
      <tr><td><strong>Model 2</strong>: MNB</td><td>78%</td><td>82%</td><td>78%</td><td>85%</td></tr>
      <tr><td><strong>Model 3</strong>: CNB</td><td>85%</td><td>85%</td><td>85%</td><td>-</td></tr>
      <tr><td><strong>Model 4</strong>: RL</td><td>87%</td><td>88%</td><td>88%</td><td>87%</td></tr>
      <tr><td><strong>Model 5</strong>: AdaBoost</td><td>80%</td><td>81%</td><td>80%</td><td>81%</td></tr>
      <tr><td><strong>Model 6</strong>: Nu-SVC</td><td>87%</td><td>88%</td><td>88%</td><td>88%</td></tr>
    </table>
  </div>

  <!-- Tableau 2 : Détection des Émotions -->
  <div>
    <h3><strong>Analyse des Émotions</strong></h3>
    <table border="1" cellpadding="5" cellspacing="0" style="width: 48%; display: inline-block;">
      <tr>
        <th>Dataset des émotions</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th>
      </tr>
      <tr><td><strong>Model 1</strong>: RNN_LSTM</td><td>93%</td><td>93%</td><td>93%</td><td>93%</td></tr>
      <tr><td><strong>Model 2</strong>: MNB</td><td>76%</td><td>80%</td><td>76%</td><td>76%</td></tr>
      <tr><td><strong>Model 3</strong>: RL</td><td>89%</td><td>89%</td><td>89%</td><td>89%</td></tr>
      <tr><td><strong>Model 4</strong>: CNB</td><td>88%</td><td>88%</td><td>88%</td><td>88%</td></tr>
      <tr><td><strong>Model 5</strong>: CNN</td><td>93%</td><td>93%</td><td>93%</td><td>93%</td></tr>
      <tr><td><strong>Model 6</strong>: AdaBoost</td><td>36%</td><td>24%</td><td>24%</td><td>21%</td></tr>
    </table>
  </div>

</div>






     
 
### Comparison de Performance des modèles 
> Une comparaison détaillée des performances des modèles pour l'analyse des sentiments et la détection des émotions est disponible [ici](https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/models%20comparaisons/interpretation%20des%20mod%C3%A8les.pdf).

### Comparaison of Models Performance
> A detailed comparison of model performance for sentiment analysis and emotion detection can be found [here](https://github.com/almasstudyjourney/BSc-Thesis-Project-Sentiment-Analysis/blob/main/models%20comparaisons/interpretation%20des%20mod%C3%A8les.pdf).


### Notes Importantes

1. Ce fichier README ne contient pas toutes les informations détaillées de notre rapport. Les techniques, les méthodes, et tous les aspects de l'étude comparative sont abordés de manière approfondie dans notre rapport complet.

2. Ce projet est une étude comparative des techniques, des approches et des méthodes utilisées pour l'analyse des sentiments. Nous avons testé plusieurs modèles et évalué leur performance en fonction de différents critères.

3. L'ajout de l'analyse des émotions dans ce projet était une décision conjointe entre ma collègue et moi-même, après avoir obtenu l'approbation de notre professeur.

4. Toutes les techniques d'évaluation des modèles, y compris la validation croisée, sont détaillées dans notre rapport. Ces méthodes ont été appliquées pour assurer la robustesse et la fiabilité de nos résultats.

5. Ce projet a été réalisé en binôme avec **Mlle Ouissal Boukoulla**. 

6. - Le **front-end** de l'application a été développé par une autre équipe, et nous avons apporté des modifications et améliorations pour l'adapter à nos besoins spécifiques.  
   - Le **back-end** a été entièrement développé par nous en utilisant **Django**, une technologie que nous avons choisie pour sa robustesse et sa flexibilité dans la gestion des données.
   - Le code de l'application sera ajouter prochainement. 

7. Nous avons ajouté une fonctionnalité permettant à l'utilisateur d'entrer du texte en français. Cette option de traduction permet de traduire les textes en anglais avant  les traiter et les analyser.
    

     

     



