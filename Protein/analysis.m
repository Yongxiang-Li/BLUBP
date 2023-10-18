% close all
% clear

if ~exist('results','var')
    load('results_protein.mat')
end
font = 18;

dmodelCML_CLP = [results(:).dmodelCML_CLP];
dmodelCML_BLUBP = [results(:).dmodelCML_BLUBP];
dmodelCCL_CLP = [results(:).dmodelCCL_CLP];
dmodelCCL_BLUBP = [results(:).dmodelCCL_BLUBP];
dmodelOCL_CLP = [results(:).dmodelOCL_CLP];
dmodelOCL_BLUBP = [results(:).dmodelOCL_BLUBP];
dmodelCML = [results(:).dmodelCML];
dmodelCCL = [results(:).dmodelCCL];
dmodelOCL = [results(:).dmodelOCL];
dmodelSPGP = [results(:).dmodelSPGP];
dmodelGPRR = [results(:).dmodelGPRR];

RMSE = [dmodelOCL_BLUBP.RMSE; dmodelCML_BLUBP.RMSE; dmodelCCL_BLUBP.RMSE; dmodelOCL_CLP.RMSE; dmodelCML_CLP.RMSE; dmodelCCL_CLP.RMSE; dmodelSPGP.RMSE; dmodelGPRR.RMSE]';
time = [[dmodelOCL_BLUBP.ptime]+[dmodelOCL.mtime];[dmodelCML_BLUBP.ptime]+[dmodelCML.mtime];[dmodelCCL_BLUBP.ptime]+[dmodelCCL.mtime];[dmodelOCL_CLP.ptime]+[dmodelOCL.mtime];[dmodelCML_CLP.ptime]+[dmodelCML.mtime];[dmodelCCL_CLP.ptime]+[dmodelCCL.mtime];dmodelSPGP.time;dmodelGPRR.time]';
figure;
subplot(1,2,1)
boxplot(RMSE,'labels', {'OCL+BLUBP'; 'CML+BLUBP'; 'CCL+BLUBP';'OCL+CLP'; 'CML+CLP'; 'CCL+CLP';'SPGP';'GPRR'})
title('(a)')
% set(gca, 'FontSize' ,font, 'XTickLabelRotation',40); 
ylabel('RMSE')
subplot(1,2,2);
boxplot(time,'labels', {'OCL+BLUBP'; 'CML+BLUBP'; 'CCL+BLUBP';'OCL+CLP'; 'CML+CLP'; 'CCL+CLP';'SPGP';'GPRR'})
title('(b)')
% set(gca, 'FontSize' ,font, 'XTickLabelRotation',40); 
ylabel('Time (s)')