# LDNFSGB:Predicting lncRNA-disease associations using network feature similarity and gradient boosting

First, we construct a feature vector to extract similarity information of different lncRNAs and diseases. Then,  an autoencoder is used to reduce the dimensionality of the feature vector. Finally, the gradientboosting classifier is used to predict lncRNA-disease associations.

#Autoencoder.py
This part is mainly to reduce the dimensionality of the sample features.

#Gradientboosting.py
It is mainly used for prediction of lncRNA-disease association.  At the same time, this part provides the performance evaluation of the model under 10-fold CV including AUC, accuracy, sensitive, precision, specificity and ROC curve drawing.


The original datasets used in this study are available from the corresponding references.