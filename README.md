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

Nous utiliserons des techniques de Traitement du Langage Naturel (TLN) comme élément clé de l'analyse avec l'apprentissage automatique. Le traitement du langage naturel permet de convertir les mots présents dans le texte en vecteurs mathématiques, nécessaires au bon fonctionnement des modèles d'apprentissage automatique. Une fois ces vecteurs mathématiques générés, ils sont transmis aux modèles d'apprentissage automatique respectifs pour la prédiction. Ces caractéristiques, ainsi que certaines nouvelles fonctionnalités créées, seront utilisées par les modèles d'apprentissage automatique et d'apprentissage profond pour générer des prédictions précises. Cette approche garantit que les modèles reçoivent les informations les plus pertinentes du texte et fournissent les meilleurs résultats possibles.  
   * **) Prétraitement des Données**
   * 
 **1)** Le processus commence par un ***nettoyage des données***, qui inclut la suppression des caractères inutiles et des symboles.
 **2)**  **Conversion de données:** Lors du prétraitement des données textuelles, il est courant de convertir tout le texte en minuscules pour éviter les problèmes liés à la casse (majuscules/minuscules). Cela permet de s'assurer que des mots identiques, mais avec des différences de casse, soient traités de la même manière. Par exemple, les mots "Hello" et "hello" seraient interprétés comme identiques après la conversion en minuscules.
     Voici un exemple de pourquoi cela est important dans le prétraitement 

     * Avant la conversion :
* Hello et hello sont considérés comme deux mots différents par un modèle.
     * Après la conversion :
* En convertissant tout le texte en minuscules, les deux mots sont normalisés en hello et seront traités comme un seul mot, ce qui améliore la cohérence de l'analyse.

  text = "Hello, how are you? I am feeling great, Hello!"
text_lower = text.lower()
print(text_lower)



     
 **2)** **Tokenisation:** Ensuite, le texte est tokenisé, c'est-à-dire découpé en unités plus petites appelées tokens (mots ou sous-mots).

**3)** **StopWords** Après la tokenisation, nous appliquons une étape de suppression des stop words, qui consiste à enlever les mots fréquents et peu informatifs, comme "et", "le", "la", "de", etc.

Ces mots ne contribuent pas à l'analyse du texte et peuvent introduire du bruit dans les modèles.


**4)** **Vectorisation** Une fois ces étapes de nettoyage effectuées, nous procédons à la vectorisation, où les mots restants sont convertis en vecteurs mathématiques à l'aide de techniques comme TF-IDF, CountVectorizer, ou Word2Vec. Ces vecteurs seront ensuite utilisés pour l'entraînement des modèles d'analyse de sentiments et d'émotions.

    

     

     



