% close all
% clear

if ~exist('results','var')
    load('Results_CCPP.mat')
end
font = 18;

dmodelML =  [results(:).dmodelML];    dmodelML_BLUP = [dmodelML(:).BLUP];
dmodelCML = [results(:).dmodelCML];    dmodelCML_CLP = [dmodelCML(:).CLP];    dmodelCML_BLUBP = [dmodelCML(:).BLUBP];
dmodelCCL = [results(:).dmodelCCL];    dmodelCCL_CLP = [dmodelCCL(:).CLP];    dmodelCCL_BLUBP = [dmodelCCL(:).BLUBP];
dmodelOCL = [results(:).dmodelOCL];    dmodelOCL_CLP = [dmodelOCL(:).CLP];    dmodelOCL_BLUBP = [dmodelOCL(:).BLUBP];

dmodelSPGP = [results(:).dmodelSPGP];
dmodelGPRR = [results(:).dmodelGPRR];

RMSE = [dmodelML_BLUP.RMSE; dmodelOCL_BLUBP.RMSE; dmodelCML_BLUBP.RMSE; dmodelCCL_BLUBP.RMSE; dmodelOCL_CLP.RMSE; dmodelCML_CLP.RMSE; dmodelCCL_CLP.RMSE; dmodelSPGP.RMSE; dmodelGPRR.RMSE]';
time = [[dmodelML.mtime]+[dmodelML_BLUP.ptime];[dmodelOCL_BLUBP.ptime]+[dmodelOCL.mtime];[dmodelCML_BLUBP.ptime]+[dmodelCML.mtime];[dmodelCCL_BLUBP.ptime]+[dmodelCCL.mtime];[dmodelOCL_CLP.ptime]+[dmodelOCL.mtime];[dmodelCML_CLP.ptime]+[dmodelCML.mtime];[dmodelCCL_CLP.ptime]+[dmodelCCL.mtime];dmodelSPGP.time;dmodelGPRR.time]';
figure;
subplot(1,2,1)
boxplot(RMSE,'labels', {'ML+BLUP';'OCL+BLUBP'; 'CML+BLUBP'; 'CCL+BLUBP';'OCL+CLP'; 'CML+CLP'; 'CCL+CLP';'SPGP';'GPRR'})
title('(a)')
% set(gca, 'FontSize' ,font, 'XTickLabelRotation',40); 
ylabel('RMSE')
subplot(1,2,2);
boxplot(time,'labels', {'ML+BLUP';'OCL+BLUBP'; 'CML+BLUBP'; 'CCL+BLUBP';'OCL+CLP'; 'CML+CLP'; 'CCL+CLP';'SPGP';'GPRR'})
title('(b)')
% set(gca, 'FontSize' ,font, 'XTickLabelRotation',40); 
ylabel('Time (s)')
% savefig('RMSE_Time_80')
