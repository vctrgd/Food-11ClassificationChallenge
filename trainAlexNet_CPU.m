% Entraînement modèle AlexNet sur FOOD11

%% Initialisation
close all
clear
clc
disp('AlexNet - Entraînement sur FOOD11...');

%% Préparation des données
[augmentedTrain, augmentedVal] = prepareData();
disp("Préparation des données terminée.");

%% Correction taille d'entrée pour AlexNet (227x227)
inputSize = [227 227];
% Si la taille n'est pas correcte, on reconstruit les datastores à partir du dossier train
if isa(augmentedTrain, 'augmentedImageDatastore')
    if ~isequal(augmentedTrain.OutputSize(1:2), inputSize)
        datasetPath = fullfile(pwd, 'train');
        imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        % Split stratifié simple (80/20)
        rng(42);
        [imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
        augmenter = imageDataAugmenter(...
            'RandXReflection',true, ...
            'RandRotation',[-10 10], ...
            'RandXTranslation',[-10 10], ...
            'RandYTranslation',[-10 10], ...
            'RandScale',[0.9 1.1]);
        augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
        augmentedVal = augmentedImageDatastore(inputSize, imdsVal);
    end
end

%% Récupération des classes
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

%% Chargement du modèle
try
    % charge AlexNet pré-entraîné
    % attention : requiert Deep Learning Toolbox Model for AlexNet Network !!
    net = alexnet();
catch
    % Si vous préférez utiliser imagePretrainedNetwork helper
    [net, ~] = imagePretrainedNetwork("alexnet");
end
disp('Modèle chargé : AlexNet');
inputSize = net.Layers(1).InputSize(1:2); % normalement [227 227]

%% Modification des couches finales
lgraph = layerGraph(net);

% Trouver la dernière fullyConnected et classification layers
fcIdx = find(arrayfun(@(x) isa(x,'nnet.cnn.layer.FullyConnectedLayer'), lgraph.Layers), 1, 'last');
classIdx = find(arrayfun(@(x) isa(x,'nnet.cnn.layer.ClassificationOutputLayer'), lgraph.Layers), 1, 'last');
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
newClassLayer = classificationLayer('Name','new_classoutput');

% Remplacer la dernière FC et la couche de classification existante
lgraph = replaceLayer(lgraph, lgraph.Layers(fcIdx).Name, newLearnableLayer);
lgraph = replaceLayer(lgraph, lgraph.Layers(classIdx).Name, newClassLayer);

% Note : AlexNet a déjà une softmax avant la classification dans la plupart des cas,
% replaceLayer ci-dessus suffit normalement.

%% Data augmentation adaptée à AlexNet inputSize
% Si prepareData() retourne des augmentedImageDatastore compatibles, conserver.
% Sinon recréer des augmentedImageDatastore en forçant inputSize.

% Si augmentedTrain existe et est un augmentedImageDatastore, redimensionner si besoin
if exist('augmentedTrain','var') && isa(augmentedTrain,'augmentedImageDatastore')
    % on ne fait rien car compatible
else
    % Recréer les datastores à partir des dossiers si nécessaire
    datasetPath = fullfile(pwd, 'train');
    imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    % Split stratifié simple (80/20)
    rng(42);
    [imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
    augmenter = imageDataAugmenter(...
        'RandXReflection',true, ...
        'RandRotation',[-10 10], ...
        'RandXTranslation',[-10 10], ...
        'RandYTranslation',[-10 10], ...
        'RandScale',[0.9 1.1]);
    augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
    augmentedVal = augmentedImageDatastore(inputSize, imdsVal);
end

%% Options d'entraînement adaptées au CPU
miniBatchSize = 32;
valFreq = max(1, floor(augmentedTrain.NumObservations / miniBatchSize));
options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 6, ...               % epochs réduites pour un temps raisonnable CPU
    'InitialLearnRate', 1e-3, ...
    'Momentum', 0.9, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedVal, ...
    'ValidationFrequency', valFreq, ...
    'ExecutionEnvironment', 'cpu', ... % forcer CPU
    'Verbose', true, ...
    'Plots', 'training-progress');

%% 8. Entraînement
trainedNet = trainNetwork(augmentedTrain, lgraph, options);

%% 9. Sauvegarde (nom explicite pour AlexNet CPU)
save('trainedAlexNet_CPU.mat', 'trainedNet', 'classes');
disp(' Modèle sauvegardé : trainedAlexNet_CPU.mat');
