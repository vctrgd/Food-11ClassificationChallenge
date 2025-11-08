%% Entraînement ResNet-18 sur FOOD11 - VERSION CORRIGÉE & PROPRE
close all; clear; clc;
disp('--- Entraînement ResNet-18 sur FOOD11 ---');

%% 1. Préparation des données (avec imdsVal pour labels)
[augmentedTrain, augmentedVal, imdsVal] = prepareData();

%% 2. Récupération des classes (directement depuis imdsVal)
classes = categories(imdsVal.Labels);
numClasses = numel(classes);
disp(['Classes détectées : ' num2str(numClasses)]);

%% 3. Chargement du modèle
net = resnet18();
lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

%% 4. Remplacement des couches finales (robuste)
fcLayer = fullyConnectedLayer(numClasses, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
classLayer = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, 'fc1000', fcLayer);           % nom fixe ResNet-18
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', classLayer);

%% 5. Options d'entraînement
miniBatchSize = 32;
options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 3, ...
    'InitialLearnRate', 5e-5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% 6. Entraînement
disp('Entraînement en cours...');
trainedNet = trainNetwork(augmentedTrain, lgraph, options);

%% 7. Évaluation (CORRECTE)
YPred = classify(trainedNet, augmentedVal);
YTrue = imdsVal.Labels;  % Vrai label, cohérent avec le split
accuracy = mean(YPred == YTrue);
disp(['Précision validation : ' num2str(accuracy*100, '%.2f') '%']);

%% 8. Sauvegarde
save('trainedResNet18_Food11_Light.mat', 'trainedNet', 'classes', '-v7.3');
disp('Modèle sauvegardé.');