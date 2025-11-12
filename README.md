# The Food-11 Classification Challenge
Ce projet consiste à concevoir et évaluer un pipeline de classification d'images sur le dataset Food-11 (16393 images, 11 classes). L'objectif est de produire un fichier JSON associant chaque image de test à une prédiction de classe compatible avec le format demandé pour le serveur d'évaluation.

## Introduction

Données
- Classes (11) : Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, Vegetable/Fruit.
- Répartition fournie : train = 13 296 images (avec répertoires par classe), test = 3 097 images (sans labels).
- Les images d'entraînement peuvent être réparties librement entre train/validation lors des expériences.

Objectifs
- Mettre en place un workflow reproductible (prétraitement → entraînement → évaluation → inférence).
- Utiliser une stratégie de transfer learning sur un réseau pré-entraîné (ex. MobileNet, ResNet, EfficientNet) adaptée aux contraintes matérielles.
- Générer un fichier JSON de prédictions respectant le format requis pour la soumission.

Méthodologie proposée
- Prétraitement :
  - Normalisation des images (échelle [0,1] ou normalisation selon le backbone choisi).
  - Redimensionnement cohérent (ex. 224×224 ou 256×256).
- Augmentation (appliquer pendant l'entraînement) :
  - Rotations faibles, translations, flips horizontaux, variations de luminosité/contraste, zoom.
- Modèle :
  - Transfer learning : charger un backbone pré-entraîné sur ImageNet, remplacer la tête par une couche fully-connected adaptée (11 classes) avec softmax.
  - Options : entraînement de la tête uniquement (initial), puis fine-tuning de quelques couches supérieures.
- Entraînement :
  - Split proposé : 80% train / 20% validation (ou cross-validation stratifiée si souhaité).
  - Early stopping sur la métrique F1 macro / validation loss.
  - Scheduler de taux d'apprentissage (ex. ReduceLROnPlateau, cosine annealing).
  - Batch size, nombre d'époques et lr à rechercher par validation.
- Évaluation :
  - Mesures : accuracy, precision, recall, F1 score macro et par classe, matrice de confusion.
  - Préserver les checkpoints et logs (TensorBoard ou équivalent).

Remarques expérimentales et bonnes pratiques
- Utiliser des seeds fixes pour reproductibilité.
- Monitorer overfitting (différence train/val) et utiliser régularisation si besoin.
- Conserver les métadonnées et scripts permettant de reproduire exactement la prédiction finale (versions de librairies, architectures, poids).
- Vérifier l'interdiction d'utiliser d'autres distributions du dataset.

Exemple minimal de pipeline (à implémenter dans le dépôt)
- data_prepare.py : création des splits et loader.
- train.py : entraînement avec sauvegarde des checkpoints.
- predict.py : inférence sur le dossier test et écriture du JSON.
- evaluate.py : calcul des métriques sur validation.

## Observations d'overfitting (ResNet18) et comparaison avec ResNet50

Symptômes observés (ResNet18)
- Courbes d'entraînement : accuracy train élevée rapidement (>90%) alors que l'accuracy validation plafonne et décline. Loss train décroît fortement mais loss validation stagne ou augmente.
- Metrices : précision et recall par classe très fluctuants en validation ; F1 macro sur validation nettement inférieur à l'entraînement.
- Comportement typique : bon fit sur exemples vus, mauvaise généralisation sur validation/test.

Causes plausibles
- Dataset limité / variabilité insuffisante par classe → modèle mémorise.
- Augmentations insuffisantes ou mal configurées.
- Hyperparamètres (lr trop élevé, trop d'époques sans early stopping).
- Absence ou faiblesse de régularisation (weight decay, dropout).
- Problème d'implantation du pipeline (fuite de données improbable mais à vérifier : shuffle, split, normalisation cohérente).
- Déséquilibre de classes non pris en compte.

Comparaison ResNet18 vs ResNet50 (observations et interprétation)
- ResNet18 (capacité réduite) :
  - Apprentissage rapide de caractéristiques de bas niveau ; dans nos expérimentations il a montré overfitting plus tôt (vraisemblablement parce que la tête entraînée converge vite et que le modèle s'adapte aux artefacts).
  - Avantage : moins coûteux en ressources, plus rapide pour itérations rapides.
- ResNet50 (capacité supérieure) :
  - Peut généraliser mieux si correctement régularisé et si fine-tuning progressif ; mais peut aussi mémoriser encore plus si on ne contrôle pas l'overfitting (plus de paramètres → plus de risque sans données/augmentation suffisantes).
  - Souvent, ResNet50 donne de meilleures performances finales sur validation si on applique un fine-tuning prudent (geler bas/backbone, entraîner tête, puis dé-geler couches supérieures progressivement).

Expérience de comparaison recommandée (plan contrôlé)
- Protocole :
  - Geler le backbone initialement ; entraîner uniquement la tête 10–20 epochs.
  - Réduire lr pour le fine-tuning (ex. lr_head = 1e-3, lr_backbone = 1e-4 ou moins).
  - Utiliser les mêmes augmentations et le même scheduler pour les deux architectures.
  - Mesurer : accuracy train/val, loss train/val, F1 macro, matrice de confusion, temps/epoch et mémoire.
- Mesures à collecter :
  - Courbes train vs val (accuracy et loss).
  - F1 macro final sur validation.
  - Overfitting index simple : (train_acc - val_acc).
- Répéter avec variations :
  - Ajouter weight decay (ex. 1e-4 → 1e-3).
  - Tester augmentations plus fortes (CutMix, MixUp, random erasing).
  - Tester dropout sur la tête (0.3–0.5).
  - Tester réduction du lr initial et scheduler (ReduceLROnPlateau).

Actions correctives prioritaires (ordre recommandé)
1. Vérifier le split et les pipelines de prétraitement pour éviter toute fuite.
2. Renforcer les augmentations (flip, rotation, brightness, CutMix/MixUp).
3. Introduire weight decay et dropout sur la tête.
4. Utiliser early stopping basé sur F1 macro validation.
5. Commencer par entraîner tête seulement, puis fine-tuner progressivement (dé-geler groupes de couches).
6. Si resurgenace de l'overfitting : réduire lr, réduire nombre d'époques, ou utiliser techniques de régularisation avancées (label smoothing, batch norm tuning).
7. Envisager l'ensemblage (ResNet18 + ResNet50) pour gains de robustesse si temps/ressources disponibles.

Interprétation pratique
- Si ResNet18 overfit malgré régularisation, préférer ResNet50 avec fine-tuning prudent (gel initial + régularisation) — ResNet50 a plus de capacité de représenter discriminants utiles si on évite la mémorisation.
- Si ressources limitées et budget d'entraînement court, travailler sur ResNet18 mais augmenter fortement les augmentations et la régularisation ; chercher à réduire le gap train/val plutôt que d'augmenter capacity.

Prochaines étapes expérimentales immédiates
- Exécuter 3 runs comparatifs contrôlés (ResNet18 vs ResNet50) avec :
  - même seed, même optimizer (AdamW ou SGD avec momentum), scheduler identique,
  - baseline d'augmentations puis version renforcée,
  - mesurer F1 macro et overfitting index.
- Documenter chaque run (hyperparams, checkpoints, courbes) pour décider la stratégie finale avant soumission.
