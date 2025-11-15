%% Initialisation
close all; clear; clc;
disp('AlexNet - Génération du fichier submissionAlexNat_CPU.json...');

%% Chargement du modèle entraîné
modelFile = 'trainedAlexNet_CPU.mat';
if ~isfile(modelFile)
    error("Modèle non trouvé, il faut exécuter trainAlexNet_CPU.m");
end

load(modelFile, 'trainedNet', 'classes');
disp('Modèle chargé : AlexNet');

%% Vérification du dossier test
testPath = fullfile(pwd, 'test');
if ~isfolder(testPath)
    error('Dossier "test" introuvable, voir README.md');
end

%% Création du datastore test
imdsTest = imageDatastore(testPath, 'IncludeSubfolders', false);
disp(['Images dans test : ' num2str(numel(imdsTest.Files))]);

%% Redimensionnement
inputSize = trainedNet.Layers(1).InputSize(1:2);
augmentedTest = augmentedImageDatastore(inputSize, imdsTest);

%% Prédiction
disp('Prédiction en cours...');
[predictedLabels, ~] = classify(trainedNet, augmentedTest);

%% Nettoyage des noms de fichiers
fileNames = imdsTest.Files;
fileNames = extractAfter(fileNames, [testPath filesep]);
fileNames = strrep(fileNames, '\', '/');
fileNames = erase(fileNames, '/');

% Tri numérique des noms
[~, idx] = sort(str2double(erase(fileNames, '.jpg')));
fileNames = fileNames(idx);
predictedLabels = predictedLabels(idx);

%% Création du dictionnaire JSON
submissionMap = containers.Map;

for i = 1:numel(predictedLabels)
    % index (0-based)
    key = string(i-1);
    submissionMap(key) = char(predictedLabels(i)); % valeur = label prédite
end

%% Conversion en JSON et sauvegarde
outputFile = 'submissionAlexNet_CPU.json';

% Suppression automatique de l'ancien fichier
if isfile(outputFile)
    delete(outputFile);
    disp('Ancien fichier submissionAlexNet_CPU.json supprimé.');
end

% Écriture du nouveau fichier
jsonStr = jsonencode(submissionMap, 'PrettyPrint', true);

fid = fopen(outputFile, 'w');
if fid == -1
    error('Impossible de créer submissionAlexNet_CPU.json');
end
fwrite(fid, jsonStr, 'char');
fclose(fid);

disp('fichier JSON généré avec succès !');
disp(["Nombre d'images : " num2str(numel(predictedLabels))]);
