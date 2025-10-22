%% Extract the FOOD11 images into arrays
disp('Preparing FOOD11 data...');

datasetPath = fullfile(pwd, 'train');  

%disp(datasetPath)
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp("Nombre total d'images : " + numel(imds.Files));
disp("Classes trouvées :");
disp(categories(imds.Labels));

% 4. Équilibrage des classes (optionnel)
%[imdsBalanced, ~] = splitEachLabel(imds, min(countEachLabel(imds).Count), 'randomized');

% 5. Division en train / validation (80/20)
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

% 6. Redimensionnement des images à une taille fixe (par exemple : 224x224)
inputSize = [224 224];  % à adapter selon le modèle utilisé plus tard

augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain);
augmentedVal   = augmentedImageDatastore(inputSize, imdsVal);

% 7. Sauvegarde des objets préparés
save('prepared_data.mat', 'augmentedTrain', 'augmentedVal');
clear imds imdsTrain imdsVal imgDataTest imgDataTrain datasetPath

disp("Préparation des données terminée.");
