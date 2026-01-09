%% ==============================================================
% AI vs Traditional IDS on NSL-KDD (MATLAB Online, CSV version)
% Requires: KDDTrain.csv, KDDTest.csv in current folder
% Produces: IDS_results_summary.csv, IDS_NSFKDD_models.mat
% ==============================================================

clear; clc; close all;

%% 1) LOAD CSVs (robust, no headers assumed)
trainFile = 'KDDTrain.csv';
testFile  = 'KDDTest.csv';
assert(isfile(trainFile) && isfile(testFile), ...
    'Cannot find KDDTrain.csv / KDDTest.csv in the current folder.');

% Raw read as data (no headers), comma-delimited
trainTbl = readtable(trainFile, 'FileType','text', 'Delimiter',',', 'ReadVariableNames',false);
testTbl  = readtable(testFile,  'FileType','text', 'Delimiter',',', 'ReadVariableNames',false);

% Drop a leading integer index col if present
if isnumeric(trainTbl{:,1}) && all(mod(trainTbl{1:min(200,height(trainTbl)),1},1)==0)
    trainTbl(:,1) = [];
end
if isnumeric(testTbl{:,1}) && all(mod(testTbl{1:min(200,height(testTbl)),1},1)==0)
    testTbl(:,1) = [];
end

% Expect 41 features + label (+ optional difficulty)
nct  = width(trainTbl);
ncte = width(testTbl);
if ~ismember(nct,[42 43]) || ~ismember(ncte,[42 43])
    error('Unexpected columns. Train=%d, Test=%d (expected 42 or 43).', nct, ncte);
end

% Name columns
if nct==42,  trainTbl.Properties.VariableNames = [compose("f%d",1:41), "label"];
else,        trainTbl.Properties.VariableNames = [compose("f%d",1:41), "label","difficulty"]; end
if ncte==42, testTbl.Properties.VariableNames  = [compose("f%d",1:41), "label"];
else,        testTbl.Properties.VariableNames  = [compose("f%d",1:41), "label","difficulty"]; end

% Drop difficulty if present
if any(strcmp(trainTbl.Properties.VariableNames,'difficulty')); trainTbl.difficulty = []; end
if any(strcmp(testTbl.Properties.VariableNames,'difficulty'));  testTbl.difficulty  = []; end

fprintf('Loaded CSVs: train=%d rows, test=%d rows.\n', height(trainTbl), height(testTbl));

%% 2) PREPROCESS (auto-detect categorical, one-hot, scale)
featTrain = trainTbl(:,1:41);
featTest  = testTbl(:,1:41);
vnames    = featTrain.Properties.VariableNames;

% auto-detect categorical/text variables in features
isCat = false(1,numel(vnames));
for i=1:numel(vnames)
    v = featTrain.(vnames{i});
    isCat(i) = iscategorical(v) || isstring(v) || ischar(v) || (iscell(v) && iscellstr(v));
end
catCols = vnames(isCat);
fprintf('Detected categorical feature columns: %s\n', strjoin(catCols, ', '));

% Labels (string)
yTrainStr = string(trainTbl.label);
yTestStr  = string(testTbl.label);

% Binary labels: 1=attack, 0=normal
yTrainBin = ~strcmpi(yTrainStr,'normal');
yTestBin  = ~strcmpi(yTestStr, 'normal');

% Multi-class numeric labels w/ consistent mapping
[yTrainMC, classList] = grp2idx(categorical(yTrainStr));
[~, yTestMC]          = ismember(categorical(yTestStr), categorical(classList));
yTestMC(yTestMC==0) = 1; % fallback if unseen label

% Build X (one-hot categoricals, z-score scaling with train stats)
[trainX, testX] = oneHotAndScale(featTrain, featTest, catCols);
fprintf('Preprocessing done. X_train: %dx%d, X_test: %dx%d\n', size(trainX,1), size(trainX,2), size(testX,1), size(testX,2));

%% 3) TRAIN & EVALUATE
results = struct([]);

% 3.1 Decision Tree (baseline)
tic;
mdlDT = fitctree(trainX, yTrainMC, 'MinLeafSize',1);
tDT = toc;
predDT = predict(mdlDT, testX);
resDT  = evaluateAll('Decision Tree', predDT, yTestMC, yTestBin);
resDT.TrainTimeSec = tDT;
results = [results; resDT];

% 3.2 SVM (RBF) via ECOC for multi-class
tic;
t = templateSVM('KernelFunction','rbf','Standardize',true);
mdlSVM = fitcecoc(trainX, yTrainMC, 'Learners', t, 'Coding','onevsone');
tSVM = toc;
predSVM = predict(mdlSVM, testX);
resSVM  = evaluateAll('SVM (RBF, ECOC)', predSVM, yTestMC, yTestBin);
resSVM.TrainTimeSec = tSVM;
results = [results; resSVM];

% 3.3 Random Forest (TreeBagger)
tic;
mdlRF = TreeBagger(150, trainX, yTrainMC, 'Method','classification', 'OOBPrediction','on');
tRF = toc;
predRF = str2double(predict(mdlRF, testX));
resRF  = evaluateAll('Random Forest (150)', predRF, yTestMC, yTestBin);
resRF.TrainTimeSec = tRF;
results = [results; resRF];

%% 4) DISPLAY & PLOTS
T = struct2table(results);
disp('Model comparison:');
disp(T(:, {'Model','Accuracy','MacroPrecision','MacroRecall','MacroF1','FPR_binary','TrainTimeSec'}));

figure('Color','w'); bar(categorical(T.Model), T.Accuracy*100);
ylabel('Accuracy (%)'); title('Accuracy (NSL-KDD)'); grid on;

figure('Color','w'); bar(categorical(T.Model), T.MacroF1*100);
ylabel('Macro-F1 (%)'); title('Macro-F1 (NSL-KDD)'); grid on;

figure('Color','w'); bar(categorical(T.Model), T.FPR_binary*100);
ylabel('FPR (%)'); title('Binary False Positive Rate (normal→attack)'); grid on;

% Confusion charts
figure('Name','DT Confusion','Color','w');
confusionchart(yTestMC, results(1).PredMC, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized', 'Title','Decision Tree');

figure('Name','SVM Confusion','Color','w');
confusionchart(yTestMC, results(2).PredMC, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized', 'Title','SVM (RBF, ECOC)');

figure('Name','RF Confusion','Color','w');
confusionchart(yTestMC, results(3).PredMC, 'RowSummary','row-normalized', 'ColumnSummary','column-normalized', 'Title','Random Forest (150)');

%% 5) SAVE
writetable(T, 'IDS_results_summary.csv');
save('IDS_NSFKDD_models.mat','mdlDT','mdlSVM','mdlRF','classList','catCols');
disp('Saved IDS_results_summary.csv and IDS_NSFKDD_models.mat');

%% ================== helper functions ==================
function [Xtr, Xte] = oneHotAndScale(trainTbl, testTbl, catCols)
    allVars  = trainTbl.Properties.VariableNames;
    isCatCol = ismember(allVars, catCols);

    trCatTbl = trainTbl(:, isCatCol);
    teCatTbl = testTbl(:, isCatCol);
    trNumTbl = trainTbl(:, ~isCatCol);
    teNumTbl = testTbl(:, ~isCatCol);

    % One-hot encode using training categories
    trCatMat = []; teCatMat = [];
    for i = 1:width(trCatTbl)
        vname = trCatTbl.Properties.VariableNames{i};
        trStr = string(trCatTbl.(vname));
        teStr = string(teCatTbl.(vname));
        cTr   = categorical(trStr);
        cats  = categories(cTr);
        cTe   = categorical(teStr, cats); % unseen -> <undefined>
        dTr   = dummyvar(cTr);
        dTe   = dummyvar(cTe);
        if size(dTe,2) < size(dTr,2), dTe(:, end+1:size(dTr,2)) = 0; end
        trCatMat = [trCatMat, dTr]; %#ok<AGROW>
        teCatMat = [teCatMat, dTe]; %#ok<AGROW>
    end

    % Numeric block
    trNumMat = table2array(varfun(@double, trNumTbl));
    teNumMat = table2array(varfun(@double, teNumTbl));

    % Concatenate and scale with TRAIN stats only
    XtrRaw = [trNumMat, trCatMat];
    XteRaw = [teNumMat, teCatMat];
    mu  = mean(XtrRaw,1);
    sg  = std(XtrRaw,0,1); sg(sg==0) = 1;
    Xtr = (XtrRaw - mu) ./ sg;
    Xte = (XteRaw - mu) ./ sg;
end

function out = evaluateAll(modelName, predMC, yMC, yBin)
    acc = mean(predMC == yMC);

    C = confusionmat(yMC, predMC);
    k = size(C,1);
    prec = zeros(1,k); rec = zeros(1,k);
    for i=1:k
        tp = C(i,i);
        fp = sum(C(:,i)) - tp;
        fn = sum(C(i,:)) - tp;
        prec(i) = tp / max(1,(tp+fp));
        rec(i)  = tp / max(1,(tp+fn));
    end
    macroPrec = mean(prec);
    macroRec  = mean(rec);
    macroF1   = 2*macroPrec*macroRec / max(1e-12,(macroPrec+macroRec));

    % Binary FPR: infer which class is 'normal' by majority among yBin==0
    normCandidates = predMC(yBin==0);
    if isempty(normCandidates), normClass = mode(predMC);
    else,                        normClass = mode(normCandidates);
    end
    predBin = predMC ~= normClass;       % 1=attack, 0=normal
    FP = sum(predBin==1 & yBin==0);
    TN = sum(predBin==0 & yBin==0);
    FPR = FP / max(1,(FP+TN));

    out = struct('Model',modelName, ...
                 'Accuracy',acc, ...
                 'MacroPrecision',macroPrec, ...
                 'MacroRecall',macroRec, ...
                 'MacroF1',macroF1, ...
                 'FPR_binary',FPR, ...
                 'TrainTimeSec',NaN, ...
                 'PredMC',predMC);
end
%% ===== ROC curves (binary: attack=1 vs normal=0) =====
% 1) Decision Tree (binary)
mdlDT_bin  = fitctree(trainX, yTrainBin, 'MinLeafSize', 1);
[~,scoreDT] = predict(mdlDT_bin, testX);              % N x 2, column 2 ~ P(attack)
[rocX_DT,rocY_DT,~,aucDT] = perfcurve(yTestBin, scoreDT(:,2), 1);

% 2) SVM RBF (binary) – get posterior scores for ROC
mdlSVM_bin = fitcsvm(trainX, yTrainBin, 'KernelFunction','rbf', ...
                     'Standardize',true, 'ClassNames',[0 1]);
mdlSVM_bin = fitPosterior(mdlSVM_bin, trainX, yTrainBin); % calibrate scores
[~,scoreSVM] = predict(mdlSVM_bin, testX);                % column 2 => P(attack)
[rocX_SVM,rocY_SVM,~,aucSVM] = perfcurve(yTestBin, scoreSVM(:,2), 1);

% 3) Random Forest (TreeBagger) – class probabilities
mdlRF_bin  = TreeBagger(150, trainX, yTrainBin, 'Method','classification', ...
                        'OOBPrediction','off');
[~,scoreRF] = predict(mdlRF_bin, testX);                  % cell labels + N x 2 probs
scoreRF = cellfun(@str2double, scoreRF);  % ensure numeric; or use second output only if numeric
if size(scoreRF,2) == 2
    sRF = scoreRF(:,2);                                   % P(attack)
else
    sRF = scoreRF;                                        % fallback if already numeric
end
[rocX_RF,rocY_RF,~,aucRF] = perfcurve(yTestBin, sRF, 1);

% Plot the three ROC curves on one figure
figure('Color','w');
plot(rocX_DT, rocY_DT, '-', 'LineWidth', 2); hold on;
plot(rocX_SVM, rocY_SVM, '-', 'LineWidth', 2);
plot(rocX_RF, rocY_RF, '-', 'LineWidth', 2);
plot([0 1],[0 1],'k:');  % chance line
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curves: Attack (positive) vs Normal (negative)');
legend({sprintf('Decision Tree (AUC = %.3f)', aucDT), ...
        sprintf('SVM RBF (AUC = %.3f)', aucSVM), ...
        sprintf('Random Forest (AUC = %.3f)', aucRF)}, ...
        'Location','southeast');
grid on; axis square;

% Save for the dissertation
saveas(gcf, 'Figure_4_4_ROC_DT_SVM_RF.png');

