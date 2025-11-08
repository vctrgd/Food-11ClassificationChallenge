%% PRÉDICTION & SOUMISSION - FOOD11 CHALLENGE (version JSON corrigée)
close all; clear; clc;
disp('=== PRÉDICTION SUR TEST SET (JSON) ===');

%% 1. Chargement du modèle entraîné
modelFile = 'trainedResNet18_Food11_Light.mat';
if ~isfile(modelFile)
    error('Modèle non trouvé ! Exécute d''abord train_resnet.m');
end

load(modelFile, 'trainedNet', 'classes');
disp('Modèle chargé : ResNet-18');

%% 2. Vérification du dossier test
testPath = fullfile(pwd, 'test');
if ~isfolder(testPath)
    error('Dossier ''test'' introuvable !');
end

%% 3. Création du datastore test
imdsTest = imageDatastore(testPath, 'IncludeSubfolders', false);
disp(['Images dans test : ' num2str(numel(imdsTest.Files))]);

%% 4. Redimensionnement
inputSize = trainedNet.Layers(1).InputSize(1:2);
augmentedTest = augmentedImageDatastore(inputSize, imdsTest);

%% 5. Prédiction (GPU si dispo)
disp('Prédiction en cours...');
[predictedLabels, ~] = classify(trainedNet, augmentedTest, 'ExecutionEnvironment', 'gpu');

%% 6. Nettoyage des noms de fichiers
fileNames = imdsTest.Files;
fileNames = extractAfter(fileNames, [testPath filesep]);
fileNames = strrep(fileNames, '\', '/');
fileNames = erase(fileNames, '/');

%% 7. Création du dictionnaire JSON avec containers.Map
submissionMap = containers.Map; % clés -> valeurs

for i = 1:numel(predictedLabels)
    key = string(i-1);                    % index (0-based)
    submissionMap(key) = char(predictedLabels(i)); % valeur = label prédite
end

%% 8. Conversion en JSON et sauvegarde
jsonStr = jsonencode(submissionMap, 'PrettyPrint', true);

fid = fopen('submission.json', 'w');
if fid == -1
    error('Impossible de créer submission.json');
end
fwrite(fid, jsonStr, 'char');
fclose(fid);

disp('✅ submission.json généré avec succès !');
disp(['Nombre d''images : ' num2str(numel(predictedLabels))]);
