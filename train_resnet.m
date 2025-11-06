%% Entraînement d’un modèle ResNet-50 par transfert sur FOOD11
% Assurez-vous que prepareData.m est dans le même dossier et exécutable

close all
clear
clc

disp('--- Entraînement ResNet-50 sur FOOD11 ---');

%% 1. Préparation ou chargement des données
if isfile('prepared_data.mat')
    disp('Chargement des données préparées...');
    load('prepared_data.mat', 'augmentedTrain', 'augmentedVal');
else
    disp('Préparation des données...');
    [augmentedTrain, augmentedVal] = prepareData();
end

%% 2. Récupération des classes
try
    % Si la version MATLAB le permet
    imdsInfo = augmentedTrain.UnderlyingDatastores{1};
    classes = categories(imdsInfo.Labels);
catch
    % Méthode alternative
    datasetPath = fullfile(pwd, 'train');
    imdsTemp = imageDatastore(datasetPath, 'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    classes = categories(imdsTemp.Labels);
end
numClasses = numel(classes);
disp("Nombre de classes détectées : " + numClasses);

%% 3. Chargement du modèle ResNet-50 pré-entraîné
net = resnet50();
disp("Modèle chargé : " + net.Name);

% Taille d’entrée du réseau (pour vérification)
inputSize = net.Layers(1).InputSize;
disp("Taille d’entrée du modèle : " + mat2str(inputSize));

%% 4. Modification des couches finales
lgraph = layerGraph(net);
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newClassLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

disp('Couches finales remplacées.');

%% 5. (Optionnel) Data augmentation
augmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandXReflection',true);

datasetPath = fullfile(pwd, 'train');
imdsTrain = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

augmentedTrain = augmentedImageDatastore(inputSize(1:2), ...
    imdsTrain, 'DataAugmentation', augmenter);

%% 6. Options d’entraînement
miniBatchSize = 32;
valFreq = floor(augmentedTrain.NumObservations / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedVal, ...
    'ValidationFrequency', valFreq, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment','auto');

%% 7. Entraînement
disp('Démarrage de l’entraînement...');
trainedNet = trainNetwork(augmentedTrain, lgraph, options);

%% 8. Évaluation sur les données de validation
disp('Évaluation sur les données de validation...');
[YPred, scores] = classify(trainedNet, augmentedVal);
YVal = augmentedVal.UnderlyingDatastores{1}.Labels; % fonctionne en R2023+, sinon on peut recharger les labels

accuracy = mean(YPred == YVal);
disp("Précision sur la validation : " + accuracy*100 + " %");

%% 9. Sauvegarde du réseau entraîné
save('trainedResNet_Food11.mat', 'trainedNet', 'classes');
disp('Modèle sauvegardé : trainedResNet_Food11.mat');
