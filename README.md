# BLUBP
Code for the paper: Yongxiang LI, Qiang ZHOU, Wei JIANG, Kwok Leung TSUI. (2023). An Optimal Composite Likelihood Estimation and Prediction for Large-scale Gaussian Process Models. IEEE Transactions on Pattern Analysis and Machine Intelligence
------------------------------------------------------------------

Abstract: 
Large-scale Gaussian process (GP) modeling is becoming increasingly important in machine learning. However, the standard modeling method of GPs, which uses the maximum likelihood method and the best linear unbiased predictor, is designed to run on a single computer, which often has limited computing power. Therefore, there is a growing demand for approximate alternatives, such as composite likelihood methods, that can take advantage of the power of multiple computers. However, these alternative methods in the literature offer limited options for practitioners because most methods focus more on computational efficiency rather than statistical efficiency. Limited accurate solutions to the parameter estimation and prediction for fast GP modeling are available in the literature for supercomputing practitioners. Therefore, this study develops an optimal composite likelihood (OCL) scheme for distributed GP modeling that can minimize information loss in parameter estimation and model prediction. The proposed predictor, called the best linear unbiased block predictor (BLUBP), has the minimum prediction variance given the partitioned data. Numerical examples illustrate that both the proposed composite likelihood estimation and prediction methods provide more accurate performance than their traditional counterparts under various cases, and an extremely close approximation to the standard modeling method is observed. 