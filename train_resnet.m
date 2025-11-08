%% Entraînement d'un modèle ResNet-18 léger sur FOOD11
% Objectif : transfert rapide et simple

close all
clear
clc
disp('--- Entraînement Light : ResNet-18 sur FOOD11 ---');

%% 1. Préparation des données

[augmentedTrain, augmentedVal] = prepareData();


%% 2. Récupération des classes
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

%% 3. Chargement du modèle ResNet-18
net = resnet18();
disp('Modèle chargé : ResNet-18');
inputSize = net.Layers(1).InputSize;

%% 4. Modification des couches finales
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
lgraph = replaceLayer(lgraph, lgraph.Layers(classIdx).Name, newClassLayer);

%% 5. Data augmentation minimale
augmenter = imageDataAugmenter('RandXReflection',true);
datasetPath = fullfile(pwd, 'train');
imdsTrain = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

augmentedTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);

%% 6. Options d'entraînement light
miniBatchSize = 32;
valFreq = floor(augmentedTrain.NumObservations / miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 3, ...              % ⚡ Seulement 3 époques
    'InitialLearnRate', 5e-5, ...    % plus bas pour stabilité
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedVal, ...
    'ValidationFrequency', valFreq, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% 7. Entraînement
disp('--- Entraînement démarré (mode rapide) ---');
trainedNet = trainNetwork(augmentedTrain, lgraph, options);

%% 8. Évaluation rapide
disp('--- Évaluation ---');
[YPred, ~] = classify(trainedNet, augmentedVal);
try
    YVal = augmentedVal.UnderlyingDatastores{1}.Labels;
catch
    datasetPath = fullfile(pwd, 'train');
    imdsVal = imageDatastore(datasetPath, 'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    YVal = imdsVal.Labels;
end

accuracy = mean(YPred == YVal);
disp(['Précision validation : ' num2str(accuracy*100, '%.2f') '%']);

%% 9. Sauvegarde
save('trainedResNet18_Food11_Light.mat', 'trainedNet', 'classes');
disp('Modèle sauvegardé : trainedResNet18_Food11_Light.mat');
