# Gender prediction

## A. General information

 - Language used: `python 3`
 - Package used: `see requirement.txt`

Please see `notebook/Data exploring.ipynb` for better understanding on how I chose **Logistic regression** for this challenge

## B. How to run
### I/ Flows

There are 3 workflows in total.
 - View performance: This will split the given dataset into 2 small ones. The first 80% of given dataset will be used as training dataset and the 20% left will be used as validation (It is because I believe this is a time series problem, hence randomly forming training & validation data set from the given data does not reflect the nature of this type of problem). The model will learn from training data and its performance on validation data will be showed.
 - Model generation: in this flow, all given data is used to train the model. Output will be put in `models` folder.
 - Test on different data set: This flow is used to test model performance on test data (which is not given). The test data should be put in `data/test_data`

#### 1/ View performance

To test model performance on given dataset, just run this command
```
python src/run.py --run_type performance_view
```

Expecting output:
```
Evalute based on test set
 - custom score:  0.785724051498234
 - f1 micro:  0.8576666666666667
 - f1 macro:  0.7890482618383696
 - accuracy score 0.8576666666666667
 - confusion matrix:
[[ 431  224]
 [ 203 2142]]
```

Note that there is nothing being created during and at the end of this step.

#### 2/ Model generation

To generate a new a model, just make sure the training data (`trainingData.csv` & `trainingLabels.csv`) is in `data/raw` folder and run this following command on project folder

```
python src/run.py --run_type test_model
```

Expecting output:
```
Successfully generate a logisic model in models folder
```

#### 3/ Test on different data set

To test the model on a different data set, you need to 2 things:
 - Put 2 files, `testData.csv` (with the same format as `trainingData.csv` file) and `testLabels.csv` (with the same format as `trainingLabels.csv` file) into `data/test_data` folder.
 - In models folder, there should be 2 files `logistic.model` and `pipeline.pkl`. If the 2 files are not exist please run `Model generation` first.

On project folder please run this following command to initiate the process.

```
python src/run.py --run_type test_model
```
Expecting output:
```
Successfully load model & pipeline
Evalute based on test set
 - custom score:  0.3456411361668384
 - f1 micro:  0.2597333333333333
 - f1 macro:  0.25852312443611297
 - accuracy score 0.2597333333333333
 - confusion matrix:
[[1645 1652]
 [9452 2251]]
```

Note that there is nothing being created during and at the end of this step.
