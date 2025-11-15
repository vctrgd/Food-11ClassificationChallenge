%% Entraînement modèle LeNet-5 sur FOOD11 (baseline)
close all
clear
clc
disp('Entraînement LeNet-5 sur FOOD11...');

%% 2. Préparation des données
[augmentedTrain224, augmentedVal224, imdsTrain, imdsVal] = prepareData();

% Adapter à la taille du réseau LeNet-5 : 32x32x3
targetSize = [32 32];
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-15 15], ...          % petites rotations réalistes
    'RandXTranslation', [-5 5], ...        % décalages horizontaux
    'RandYTranslation', [-5 5], ...        % décalages verticaux
    'RandXScale', [0.9 1.1], ...           % zoom léger
    'RandYScale', [0.9 1.1]);          % éclairage variable
augmentedTrain = augmentedImageDatastore(targetSize, imdsTrain, ...
    'DataAugmentation', augmenter);
augmentedVal = augmentedImageDatastore(targetSize, imdsVal);

%% 3. Récupération des classes
% On peut obtenir les classes à partir d'une image d'exemple
sampleImage = read(augmentedTrain);
reset(augmentedTrain); % on remet le pointeur au début
% Pour connaître les classes, on recharge juste le dossier train :
imdsTemp = imageDatastore(fullfile(pwd, 'train'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imdsTemp.Labels);
numClasses = numel(classes);
disp(['Nombre de classes détectées : ' num2str(numClasses)]);

%% 4. Définition du modèle LeNet-5 from scratch
inputSize = [32 32 3];
layers = [
    imageInputLayer(inputSize, 'Name', 'input')

    convolution2dLayer(5, 6, 'Padding', 2, 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')  % ← AJOUTER
    reluLayer('Name', 'relu1') 
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(5, 16, 'Padding', 0, 'Name', 'conv2') % Conv2: 5x5, 16 maps
    batchNormalizationLayer('Name', 'bn2')  % ← AJOUTER
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(5, 120, 'Padding', 0, 'Name', 'conv3') % Conv3: 5x5, 120 maps
    reluLayer('Name', 'relu3')

    fullyConnectedLayer(84, 'Name', 'fc1') % FC1: 84 units
    reluLayer('Name', 'relu4')
    %dropoutLayer(0.6,'Name','dropout1')


    fullyConnectedLayer(numClasses, 'Name', 'fc2') % FC2: adapté à tes 11 classes
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

lgraph = layerGraph(layers); % Crée le graph des layers



%% 6. Options d'entraînement

miniBatchSize = 64;
valFreq = 40;
options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-3, ... 
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedVal, ...
    'ValidationFrequency', valFreq, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

%% 7. Entraînement
trainedNet = trainNetwork(augmentedTrain, lgraph, options);


%% 8.Évaluation rapide
disp('Évaluation en cours...');
[YPred, ~] = classify(trainedNet, augmentedVal);
YVal = imdsVal.Labels; % Utilise imdsVal extrait en section 2
accuracy = mean(YPred == YVal);
disp(['Précision validation : ' num2str(accuracy*100, '%.2f') '%']);

% Matrice de confusion
figure;
cm = confusionchart(YVal, YPred);
cm.Title = 'Matrice de confusion - LeNet-5';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';


%% Sauvegarde
save('trainedLeNet5_Food11.mat', 'trainedNet', 'classes', '-v7.3');
disp('Modèle sauvegardé : trainedLeNet5_Food11.mat');