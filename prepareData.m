function [augmentedTrain, augmentedVal] = prepareData()
%PREPAREDATA Prépare les données FOOD11 pour l'entraînement et la validation.
%
% Retourne :
%   augmentedTrain : datastore d’images augmentées pour l'entraînement
%   augmentedVal   : datastore d’images augmentées pour la validation

    disp('Preparing FOOD11 data...');

    datasetPath = fullfile(pwd, 'train');  

    imds = imageDatastore(datasetPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    disp("Nombre total d'images : " + numel(imds.Files));
    disp("Classes trouvées :");
    disp(categories(imds.Labels));

    % Division en train / validation
    [imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

    % Taille d’entrée (RegNet/ResNet → 224x224)
    inputSize = [224 224];

    % Création des datastores augmentés
    augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain);
    augmentedVal   = augmentedImageDatastore(inputSize, imdsVal);

    % Sauvegarde optionnelle
    save('prepared_data.mat', 'augmentedTrain', 'augmentedVal');

    disp("Préparation des données terminée.");
end
