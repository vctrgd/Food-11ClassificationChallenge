%% PRÉDICTION & SOUMISSION - FOOD11 CHALLENGE
% Charge le modèle entraîné et génère submission.csv
close all; clear; clc;
disp('=== PRÉDICTION SUR TEST SET ===');

%% 1. Chargement du modèle entraîné
modelFile = 'trainedResNet18_Food11_Light.mat';
if ~isfile(modelFile)
    error('Modèle non trouvé ! Exécute d''abord train_resnet.m');
end

load(modelFile, 'trainedNet', 'classes');
disp('Modèle chargé : trainedResNet18_Food11_Light.mat');
disp(['Classes : ' strjoin(string(classes), ', ')]);

%% 2. Vérification du dossier test
testPath = fullfile(pwd, 'test');
if ~isfolder(testPath)
    error('Dossier ''test'' introuvable ! Assure-toi qu''il existe.');
end

%% 3. Création du datastore test
imdsTest = imageDatastore(testPath, ...
    'IncludeSubfolders', false);  % images directement dans test/

disp(['Images dans test : ' num2str(numel(imdsTest.Files))]);

%% 4. Redimensionnement (comme à l'entraînement)
inputSize = trainedNet.Layers(1).InputSize(1:2);
augmentedTest = augmentedImageDatastore(inputSize, imdsTest);

%% 5. Prédiction
disp('Prédiction en cours...');
[predictedLabels, scores] = classify(trainedNet, augmentedTest);
[~, ~, ~, confidence] = classify(trainedNet, augmentedTest);  % max prob

%% 6. Nettoyage des noms de fichiers
fileNames = imdsTest.Files;
fileNames = extractAfter(fileNames, [testPath filesep]);  % enlève chemin
fileNames = strrep(fileNames, '\', '/');
fileNames = erase(fileNames, '/');

%% 7. Création de submission.csv
submission = table(fileNames, categorical(predictedLabels), confidence, ...
    'VariableNames', {'filename', 'label', 'confidence'});

submission = sortrows(submission, 'filename');  % tri obligatoire

writetable(submission, 'submission.csv');
disp('submission.csv généré !');
disp(['Lignes : ' num2str(height(submission))]);
disp('Prêt à soumettre !');