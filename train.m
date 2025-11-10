%% Entraînement modèle ResNet-18 sur FOOD11
close all
clear
clc
disp('--- Entraînement Light : ResNet-18 sur FOOD11 ---');

%% 1. Vérification du GPU
try
    gpu = gpuDevice(); % Sélection du GPU actif
    disp([' GPU détecté : ' gpu.Name]);
catch ME
    warning("Aucun GPU détecté. Le code s'exécutera sur CPU.");
end

%% 2. Préparation des données
[augmentedTrain, augmentedVal] = prepareData();

%% 3. Récupération des classes
try
    imdsInfo = augmentedTrain.UnderlyingDatastores{1};
    classes = categories(imdsInfo.Labels);
catch
    datasetPath = fullfile(pwd, 'train');
    imdsTemp = imageDatastore(datasetPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    classes = categories(imdsTemp.Labels);
end
numClasses = numel(classes);
disp(['Nombre de classes détectées : ' num2str(numClasses)]);

%% 4. Chargement du modèle ResNet-18
[net,netclassNames] = imagePretrainedNetwork("resnet18");
disp('Modèle chargé : ResNet-18');
inputSize = net.Layers(1).InputSize(1:2);

%% 5. Modification des couches finales
lgraph = layerGraph(net);

% Dernière couche fullyConnected
fcIdx = find(arrayfun(@(x) isa(x,'nnet.cnn.layer.FullyConnectedLayer'), lgraph.Layers), 1, 'last');
classIdx = find(arrayfun(@(x) isa(x,'nnet.cnn.layer.ClassificationOutputLayer'), lgraph.Layers), 1, 'last');

newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
newClassLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph, lgraph.Layers(fcIdx).Name, newLearnableLayer);

% Remplacer/ajouter la softmax et la classification
lgraph = replaceLayer(lgraph,'prob',softmaxLayer('Name','new_softmax'));
lgraph = addLayers(lgraph,newClassLayer);

% Connecter la softmax à la nouvelle couche de classification
lgraph = connectLayers(lgraph,'new_softmax','new_classoutput');

%% 6. Data augmentation minimale
%augmenter = imageDataAugmenter('RandXReflection',true);
%datasetPath = fullfile(pwd, 'train');
%imdsTrain = imageDatastore(datasetPath, ...
%    'IncludeSubfolders', true, ...
%   'LabelSource', 'foldernames');
%augmentedTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
%   'DataAugmentation', augmenter);

%% 7. Options d'entraînement (GPU forcé)
miniBatchSize = 64;
%valFreq = floor(augmentedTrain.NumObservations / miniBatchSize);
valFreq=30;
options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 3, ...              % Justifier nombres epochs
    'InitialLearnRate', 5e-5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedVal, ...
    'ValidationFrequency', valFreq, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

%% 8. Entraînement
trainedNet = trainNetwork(augmentedTrain, lgraph, options);

%% 9. Sauvegarde
save('trainedResNet18_Food11_Light.mat', 'trainedNet', 'classes');
disp(' Modèle sauvegardé : trainedResNet18_Food11_Light.mat');
