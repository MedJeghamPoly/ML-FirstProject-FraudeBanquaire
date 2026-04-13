---
title: Détection de fraude par carte bancaire à l’aide de l’apprentissage automatique
lang: fr
---

<div align="center">

# Détection de fraude par carte bancaire à l’aide de l’apprentissage automatique

**Rapport de projet**

**Nom de l’étudiant :** [Votre nom]

**Date :** 1<sup>er</sup> avril 2026

**Institution :** Polytech — Machine Learning (Semestre 4)

</div>

<div style="page-break-after: always;"></div>

## Avant-propos

Ce mémoire synthétise une chaîne complète de classification supervisée (prétraitement, SMOTE, comparaison de modèles, optimisation du Random Forest, évaluation sur jeu de test). Les notebooks correspondants sont `notebooks/01_EDA.ipynb` et `notebooks/02_Modeling.ipynb` ; les métriques tabulées proviennent de `outputs/model_results.csv`.

---

## 1. Introduction

La fraude par carte bancaire expose les établissements de paiement à des pertes financières, à un risque réputationnel et à des exigences réglementaires croissantes ; face au volume transactionnel, le contrôle humain exhaustif est devenu irrecevable en coût et en délai. L’apprentissage automatique permet de hiérarchiser les alertes et d’automatiser une partie du filtrage, à condition d’accepter que la rareté extrême de la classe « fraude » rend l’évaluation naïve par l’exactitude globale trompeuse.

**Ce projet** vise à comparer, sur un jeu de données tabulaire anonymisé, plusieurs classifieurs binaires dans un protocole reproductible — de l’exploration à la sélection du modèle final — et à interpréger les résultats tant sur le plan statistique que sur celui des arbitrages opérationnels. **Sur cette base**, la section suivante précise la nature des variables et les contraintes d’interprétation qu’impose l’anonymisation.

---

## 2. Description du jeu de données

Le corpus `creditcard.csv`, usité en recherche appliquée pour la détection de fraude, recense environ **285 000** transactions par carte enregistrées en **septembre 2013** pour des porteurs européens : une ligne par opération, **trente** prédicteurs numériques et la cible **Class** (0 : légitime, 1 : fraude). Les champs **Time** (secondes écoulées depuis la première transaction) et **Amount** (montant) sont conservés à l’échelle d’origine ; les attributs **V1–V28** proviennent en principe d’une **analyse en composantes principales (PCA)** appliquée à des caractéristiques brutes confidentielles. Cette discrétisation protège le secret métier mais **retire toute lecture causale directe** des importances de variables : l’on mesure l’influence de **composantes**, non de comportements clients nommables.

**Ainsi circonscrit**, le problème se formalise comme une discrimination binaire supervisée entre distributions de points dans ℝ³⁰. **Il convient maintenant** d’examiner empiriquement la forme de ces distributions — et surtout le déséquilibre des labels — avant toute modélisation.

---

## 3. Exploration des données (EDA)

L’analyse exploratoire confirme un **déséquilibre sévère** : la fraude occupe une fraction de pour cent des lignes, de sorte qu’un classifieur majoritaire trivial afficherait une exactitude élevée tout en ignorant la classe d’intérêt. Les graphiques de répartition des classes rendent visible cette asymétrie ; les distributions de **Time**, **Amount** et d’un sous-ensemble de **V*** révèlent des queues lourdes et des écarts de médiane ou de dispersion entre classes, même si l’ACP masque la lecture intuitive des effets. La **matrice de corrélation** linéaire signale des redondances partielles entre composantes, ce qui justifie — sans la suffire — l’usage de modèles non strictement linéaires capables de capturer des interactions.

**En résumé**, l’EDA impose de **décrocher** la métrique de performance du seul taux de succès global. **La phase suivante** formalise le prétraitement qui stabilise l’échelle des entrées et isole un jeu de test statistiquement honnête.

---

## 4. Prétraitement des données

Sur la version standard du fichier, l’absence quasi totale de valeurs manquantes facilite le contrôle qualité ; le pipeline conserve néanmoins une imputation médiane défensive pour préserver la robustesse du code. **Aucune variable catégorielle** n’exige d’encodage ici. La **standardisation** (moyenne nulle, variance unitaire) est **ajustée sur le seul sous-échantillon d’apprentissage** puis appliquée au test, ce qui évite toute fuite d’information vers les paramètres du scaler ; elle est indispensable pour la régression logistique et harmonise l’échelle des montants avec celle des composantes PCA. Le **partitionnement 80/20 stratifié** sur **Class** garantit une proportion de fraudes comparable dans l’entraînement et le test, condition minimale pour une estimation non biaisée du rappel et de la précision sur la minorité.

**Une fois ces garde-fous posés**, la question devient celle du **traitement du déséquilibre** en phase d’apprentissage sans dénaturer l’évaluation finale.

---

## 5. Gestion des données déséquilibrées

Le déséquilibre déplace l’optimum apparent des fonctions de perte classiques vers la classe majoritaire ; sans correction, les frontières de décision restent **conservatrices** au regard de la fraude. **SMOTE** (Synthetic Minority Over-sampling Technique) synthétise des exemplaires minoritaires par interpolation locale dans l’espace des caractéristiques, ce qui **redistribue** la densité vue par l’algorithme pendant l’**entraînement** uniquement. Dans l’implémentation retenue, SMOTE est intégré au **pipeline** après normalisation et **n’altère pas** le jeu de test, dont la prévalence de fraude demeure réaliste. Cette stratégie peut améliorer le rappel et le compromis F1 ; elle comporte toutefois un **enjeu théorique** : les points artificiels ne sont pas des transactions observées et peuvent **lisser** des régions où classes se chevauchent, d’où le maintien impératif d’une validation sur données non suréchantillonnées et, idéalement, d’une analyse de sensibilité au choix du rééchantillonneur.

**Après avoir posé ce compromis méthodologique**, on peut détailler les hypothèses algorithmiques des quatre familles comparées.

---

## 6. Modélisation

Tous les modèles partagent la même enveloppe : **normalisation → SMOTE → estimateur**. La **régression logistique** postule une séparation (quasi) linéaire en probabilité ; elle est parcimonieuse et auditables en coefficients, mais rigide si les frontières sont fortement non linéaires. L’**arbre de décision** segmente récursivement l’espace par seuils et capture des interactions locales au prix d’une variance élevée sans régularisation adéquate. La **forêt aléatoire** moyenne des arbres décorrélés ; elle excelle souvent sur données tabulaires et fournit des indicateurs d’importance — ici ses hyperparamètres ont été explorés par **GridSearchCV** avec objectif **F1**, puis le modèle a été réentraîné sur la totalité du train. Le **gradient boosting** (**XGBoost** dans les résultats exportés) enchaîne additivement des estimateurs faibles corrigeant les erreurs résiduelles ; il est compétitif mais sensible au réglage et au coût calculatoire.

**Disposant ainsi d’un panel méthodologique cohérent**, il reste à **fixer le vocabulaire d’évaluation** et à choisir les indicateurs alignés sur le risque de fraude.

---

## 7. Évaluation des modèles

Sur le jeu de test, cinq indicateurs ont été calculés. L’**exactitude** agrège corrects et incorrects sans pondérer la rareté ; elle reste **citée** pour la transparence mais ne saurait fonder seule le jugement. La **précision** sur la fraude mesure la fiabilité des alertes positives ; le **rappel** mesure la part de fraudes détectées. Le **F1** en est la moyenne harmonique : il **pénalise** les modèles qui maximisent un seul des deux termes — par exemple un rappel quasi maximal obtenu en déclenchant des milliers de fausses alertes (**précision** effondrée), configuration fréquente avec régression logistique déséquilibrée sans seuil métier. La **ROC-AUC** synthétise la capacité de rangement des scores entre classes, **indépendamment** d’un seuil de décision fixé.

**Muni de ces définitions**, les résultats empiriques permettent une hiérarchisation argumentée des algorithmes.

---

## 8. Résultats et comparaison

Les performances test (arrondies) issues de `outputs/model_results.csv` sont les suivantes.

| Modèle                       | Exactitude | Précision (fraude) | Rappel (fraude) | F1 (fraude) | ROC-AUC |
|-----------------------------|-----------:|-------------------:|----------------:|------------:|--------:|
| Régression logistique       | 97,4 %     | 5,8 %              | 91,8 %          | 0,109       | 0,971   |
| Arbre de décision           | 98,7 %     | 9,7 %              | 82,7 %          | 0,174       | 0,869   |
| Forêt aléatoire (optimisé)  | 99,9 %     | 80,4 %             | 79,6 %          | **0,800**   | **0,978** |
| Gradient boosting (XGBoost) | 99,8 %     | 45,2 %             | 86,7 %          | 0,594       | 0,976   |

La **forêt aléatoire optimisée** domine nettement au **F1** (0,80) tout en affichant la **ROC-AUC** la plus élevée (0,978) : précision et rappel sur la fraude se situent en ordre de grandeur comparable (~80 %), ce qui distingue ce modèle des approches polarisées. La régression logistique illustre le **piège classique** : rappel élevé (91,8 %) assorti d’une précision de 5,8 % — niveau d’alertes intenable opérationnellement. XGBoost maintient un rappel élevé (86,7 %) et une excellente capacité de rangement (AUC 0,976), mais une précision modérée (45,2 %) retient le F1 à 0,594 sur ce découpage. **Au vu de ces résultats**, la sélection du Random Forest repose sur un **compromis explicitement équilibré** entre coût des faux positifs et des faux négatifs, plutôt que sur un rappel « à tout prix ». **Les figures associées** permettent de vérifier visuellement ces compromis.

---

## 9. Lecture des visualisations obtenues

La **matrice de confusion** ventile les décisions correctes et les deux types d’erreur sur les classes légitime et fraude ; elle matérialise le bilan qualitatif qu’un taux d’exactitude global tend à masquer. La **courbe ROC** relie les taux de faux positifs et de vrais positifs lorsque le seuil varie : son aire (**ROC-AUC**) résume la qualité intrinsèque du **score** produit par le modèle. L’histogramme d’**importance des variables** hiérarchise les entrées selon le critère interne du modèle choisi (impureté pour les arbres) ; compte tenu de la PCA, il s’agit d’un guide de **surveillance technique** (« quelles composantes pèsent le plus dans le score ») plutôt que d’une politique produit immédiate.

**Dès lors**, il est utile de **retraduire** ces quantités dans un langage de risque et de coûts.

---

## 10. Interprétation métier

Réduire la fraude non détectée limite les pertes directes et la charge contentieuse ; réduire les faux positifs préserve la fluidité du commerce et la confiance dans le dispositif antifraude. Les **faux négatifs** coûtent typiquement par transaction frauduleuse échappée ; les **faux positifs** externalisent un coût sur la relation client et sur les équipes d’analyse. Aucun score F1, même élevé, ne **substitue** à un arbitrage explicite des coûts relatifs : en production, le **seuil** sur la probabilité prédite devrait être choisi par optimisation sous contraintes opérationnelles, en superposition avec des règles métier et, le cas échéant, une détection en flux.

**Enfin**, toute généralisation des chiffres obtenus **doit être tempérée** par les limites du dispositif expérimental.

---

## 11. Limites

**Données.** Observations datées de 2013 : les schémas de fraude et les profils de risque ont évolué ; la **validité externe** temporelle est donc **incertaine**. L’anonymisation PCA interdit la contre-expertise sémantique des `V*`. L’évaluation par **coupure transversale** stratifiée ne reproduit pas strictement une validation **chronologique** (train passé / test futur), pourtant plus proche du risque réel en autorisation continue.

**Méthodes et inférence.** SMOTE modifie la distribution d’entraînement ; son bien-fondu dépend de la géométrie locale des classes et n’équivaut pas à l’observation de nouvelles fraudes réelles. L’optimisation du Random Forest sur un **sous-échantillon** d’apprentissage pour le grid search introduit une approximation ; le test unique, bien que 20 % du corpus, ne fournit **pas** d’intervalle de confiance asymptotique sur les métriques. Les forêts et le boosting restent des modèles **post-hoc** à expliquer par outils dédiés si l’exigence réglementaire l’impose.

**Ces réserves posées**, la conclusion ne doit pas **surinterpréter** un classement ponctuel de modèles.

---

## 12. Conclusion

Sur le protocole implémenté — pipeline avec standardisation et SMOTE réservé à l’apprentissage, comparaison de quatre familles et sélection sur **F1** avec appoint **ROC-AUC** — la **forêt aléatoire optimisée** apparaît comme le candidat le plus convaincant, avec un **F1** de **0,800** et une **ROC-AUC** de **0,978** sur le jeu de test. Ce résultat est **interne** au jeu et au découpage retenus : il atteste d’une **séparation statistique** exploitable entre classes sur ce corpus anonymisé, non d’une **preuve** de performance opérationnelle en production.

**Analyse critique.** La discordance entre **AUC élevée** et **F1 modéré** pour certains modèles (régression logistique) rappelle qu’un bon **score** n’implique pas un **seuil** par défaut adapté au métier — d’où l’illusion d’un fort rappel si l’on accepte une précision proche du bruit. Inversement, le **F1** du Random Forest suggère un point de fonctionnement plus **`balanced`**, mais demeure **conditionné** par SMOTE et par l’absence de coûts asymétriques explicites dans la fonction objectif finale. **En dernière analyse**, ce travail illustre moins une « solution » close qu’une **méthodologie de décision sous incertitude** : calibrage, suivi de dérive, validation temporelle, explicabilité (**SHAP**), et ingénierie de seuils devraient précéder toute généralisation institutionnelle.

**Pistes méthodologiques.** Outre l’affinage poussé du boosting et l’exploration de réseaux profonds sur données tabulaires là où le gain marginal le justifie, les travaux futurs gagneraient à intégrer explicitement les **coûts** dans l’optimisation et à déployer une **surveillance continue** des distributions d’entrée et de sortie — étapes sans lesquelles même un modèle initialement « meilleur » au sens du F1 **ne demeure** pas fiable dans le temps.

---

*Export PDF : traitement de texte ou `pandoc Rapport_Detection_Fraude_ML.md -o rapport.pdf`. Remplacez « [Votre nom] » sur la page de titre.*
