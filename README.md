# PracticalMachineLearning

Based on a dataset provide by HAR [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) we will try to train a predictive model to predict what exercise was performed using a dataset with 159 features

The analysis is divided into two main sections:

* Exploratory data analysis
* Train and build model

The EDA section will reveal several major findings: For instance, the course's training set is a reduced version of the study's original data. More importantly, it is not even necessary to build and train a prediction model in order to achieve 100% accuracy on the testing set. Simple EDA techniques and a straightforward look up function are enough to create the correct submission files.

In the second section we will train and build a random forest model. We will cross validate the model in order to report an estimate of the out of sample error. The model will not be used to make the test set predictions because of the major drawbacks in regards to the test set structure revealed during the exploratory analysis. 
