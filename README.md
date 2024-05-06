# URL Phishing | REMLA Team 5
Simple ML application for detection of Phishing URLs. This repository contains all code to train a new model and test its performance. For installation and usage, see the sections below. The documentation at the end describes the implementation of this repository.

Note that this project only supports python versions 3.9 and 3.10.

## Installation

a) Clone repo.

```
$ git clone git@github.com:REMLA24-Team-5/Model-Training.git
$ cd REMLA-team5
```

b) Install dependencies.

```
$ python -m venv venv
$ source venv/bin/activate
$ pipx install poetry
$ poetry install
```
If you do not have pipx installed, other installation possibilities can be found in the poetry documentation https://python-poetry.org/docs/.

c) create data and output directory

```
$ mkdir output
$ mkdir data
```

d) Pull the pipeline using dvc from Google Drive

```
$ dvc pull
```

## Usage
a) After implementing changes, run the pipeline and push changes, if any, to the remote

```
$ dvc repro
$ dvc push
```

b) To run experiments and see metrics, do either of the following commands

* Show metrics after dvc repro:
```
$ dvc repro
$ dvc metrics show
```
* Or run an experiment using the following steps:
1. Run the pipeline and save experiment results.
 ```
$ dvc exp run
$ dvc metrics show
```
2. See the difference.
 ```
$ dvc metrics diff
```
3. If you change something, you can check the improved or decreased performance by running a new experiment.
 ```
$ dvc exp run
```
4. Check the experiment log:
```
$ dvc exp show
```

c) To run code quality statistical analysis:
```
pylint src
```

d) To run code security analysis:
```
bandit -c bandit.yaml -r .
```

## Documentation
### Project best practices
The team has chosen to use the cookie cutter template for [data science](https://drivendata.github.io/cookiecutter-data-science/) to ensure correct and reproducible code. Moreover, using a popular file structure for such project will make understanding and finding the code easier for people not involved in this particular project. Except for refactoring the code to this template, the team has also reduced all code to include only its main functionality. Poetry ( https://python-poetry.org/ ) guarantees that the exact same dependencies and sub-dependencies are installed in different machines. The dataset is stored on google drive and is automatically downloaded as part of the pipeline. All exploratory code is in a separate notebook in the notebooks folder.

### Pipeline Management
DVC is used for data version control and pipeline management. All different stages of the pipeline can be run with 'dvc repro'. Additionally, DVC is used to report metrics and to keep track of different experiments, models and cached data. Given the relatively simple nature of the model outputs, i.e. it uses two classes, the team has chosen to report the classification scores (multiple metrics such as precision, recall and f1-score) and also include the confusion matrix. These metrics form an all encompassing overview of the model performance, and allows for critical comparison between different models.

### Code Quality
The project implements different ways to display code quality information, considers multiple linters, critically analyses linter rules, and proposes new missing ML rules.

Pylint is used to enforce good code quality by settings certain rules. We use DSLinter (https://pypi.org/project/dslinter/) which is an extension for pylint, specifically for machine learning development. DSLinter also provides a customized .pylintrc file which defines all the linter rules. We are using this file except we removed all linter options for pytorch since we are not using the library. The .pylintrc provided first removes all basic pylint rules since standard naming conventions are counterintuitive to machine learning conventions (snake_case vs X_train for example). The rules then instantiated and used are:

```
import: check for missing imports and if existing import use correct naming conventions.

unnecessary-iteration-pandas: Vectorized solutions are preferred over iterators for DataFrames.

unnecessary-iteration-tensorflow:   Augment assignment in the loop can be replaced with vectorized solution in TensorFlow APIs.    

nan-numpy: Values cannot be compared with np.nan

chain-indexing-pandas: Chain indexing is considered bad practice in pandas code and should be avoided. If chain indexing is used on a pandas dataframe, the rule is violated.

datatype-pandas: Datatype should be set when a dataframe is imported from data to ensure the data formats are imported as expected. If the datatype is not set when importing, the rule is violated.

column-selection-pandas: Column should be selected after the dataframe is imported for better elaborating what to be expected in the downstream.

merge-parameter-pandas:  Parameters 'how', 'on' and 'validate' should be set for merge operations to ensure the correct usage of merging.

inplace-pandas: Operations on DataFrames return new DataFrames, and they should be assigned to a variable. Otherwise the result will be lost, and the rule is violated. 

dataframe-conversion-pandas: For dataframe conversion in pandas code, use .to_numpy() instead of .values. 

scaler-missing-scikitlearn: Check whether the scaler is used before every scaling-sensitive operation in scikit-learn codes. 

hyperparameters-scikitlearn: For scikit-learn learning algorithms, some important hyperparameters should be set.

hyperparameters-tensorflow: or neural network learning algorithm, some imporatnt hyperparameters should be set, such as learning rate, batch size, momentum and weight decay.

memory-release-tensorflow: If a neural network is created in the loop, and no memory clear operation is used, the rule is violated.

randomness-control-numpy: The np.seed() should be used to preserve reproducibility in a machine learning program.

randomness-control-scikitlearn: For reproducible results across executions, remove any use of random_state=None in scikit-learn estimators.


randomness-control-tensorflow:  The tf.random.set_seed() should be used to preserve reproducibility in a Tensorflow program.

missing-mask-tensorflow: If log function is used in the code, check whether the argument value is valid.

tensor-array-tensorflow:  Use tf.TensorArray() for growing array in the loop.

pipeline-not-used-scikitlearn: All scikit-learn estimators should be used inside Pipelines, to prevent data leakage between training and test data.

dependent-threshold-scikitlearn: If threshold-dependent evaluation(e.g., f-score) is used in the code, check whether threshold-indenpendent evaluation(e.g., auc) metrics is also used in the code.

dependent-threshold-tensorflow: If threshold-dependent evaluation(e.g., f-score) is used in the code, check whether threshold-indenpendent evaluation(e.g., auc) metrics is also used in the code.
```

The result from running pylint on the src code:
```
Report
======
139 statements analysed.

Statistics by type
------------------

+---------+-------+-----------+-----------+------------+---------+
|type     |number |old number |difference |%documented |%badname |
+=========+=======+===========+===========+============+=========+
|module   |9      |9          |=          |100.00      |0.00     |
+---------+-------+-----------+-----------+------------+---------+
|class    |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|method   |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|function |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+



Raw metrics
-----------

+----------+-------+------+---------+-----------+
|type      |number |%     |previous |difference |
+==========+=======+======+=========+===========+
|code      |159    |59.55 |159      |=          |
+----------+-------+------+---------+-----------+
|docstring |33     |12.36 |33       |=          |
+----------+-------+------+---------+-----------+
|comment   |22     |8.24  |22       |=          |
+----------+-------+------+---------+-----------+
|empty     |53     |19.85 |53       |=          |
+----------+-------+------+---------+-----------+



Duplication
-----------

+-------------------------+------+---------+-----------+
|                         |now   |previous |difference |
+=========================+======+=========+===========+
|nb duplicated lines      |0     |0        |0          |
+-------------------------+------+---------+-----------+
|percent duplicated lines |0.000 |0.000    |=          |
+-------------------------+------+---------+-----------+



Messages by category
--------------------

+-----------+-------+---------+-----------+
|type       |number |previous |difference |
+===========+=======+=========+===========+
|convention |0      |0        |0          |
+-----------+-------+---------+-----------+
|refactor   |0      |0        |0          |
+-----------+-------+---------+-----------+
|warning    |0      |0        |0          |
+-----------+-------+---------+-----------+
|error      |0      |0        |0          |
+-----------+-------+---------+-----------+



Messages
--------

+-----------+------------+
|message id |occurrences |
+===========+============+




--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)

```

The project is configured to be able to run the code security scanner Bandit. One line in the preprocessing of data, more specifically in the Tokenization of the input data raised an issue with Bandit. This is because Bandit scans for variables with the string "token" included in the name to check for possible hardcoded passwords. In the ML context, tokens more often than not do not refer to passwords but rather word tokens in Tokenization of input text. Thus, this line of code is skipped when running Bandit.

Bandit run results:

Test results:
        No issues identified.

Code scanned:
        Total lines of code: 174
        Total lines skipped (#nosec): 1

Run metrics:
        Total issues (by severity):
                Undefined: 0
                Low: 0
                Medium: 0
                High: 0
        Total issues (by confidence):
                Undefined: 0
                Low: 0
                Medium: 0
                High: 0
Files skipped (0):