function [augmentedTrain, augmentedVal, imdsVal] = prepareData()
%PREPAREDATA Prépare les données FOOD11 pour l'entraînement et la validation.
%
% Retourne :
%   augmentedTrain : datastore d'images augmentées pour l'entraînement
%   augmentedVal   : datastore d'images augmentées pour la validation

disp('Préparation des données FOOD11...');
    datasetPath = fullfile(pwd, 'train');
    
    imds = imageDatastore(datasetPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    disp(['Total images: ' num2str(numel(imds.Files))]);
    
    [imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');
    
    inputSize = [224 224];
    
    % Augmentation uniquement sur train
    augmenter = imageDataAugmenter('RandXReflection', true);
    
    augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
        'DataAugmentation', augmenter);
    
    % Même taille, mais PAS d'augmentation sur val (ou même aug si TTA)
    augmentedVal = augmentedImageDatastore(inputSize, imdsVal);
    
    disp('Préparation terminée.');
end