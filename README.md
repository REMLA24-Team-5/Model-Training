# URL Phishing | REMLA Team 5
Simple ML application for detection of Phishing URLs. This repository contains all code to train a new model and test its performance. For installation and usage, see the sections below. The documentation at the end describes the implementation of this repository.

## Installation
a) Clone repo.

```
$ git clone git@github.com:REMLA24-Team-5/REMLA-team5.git
$ cd REMLA-team5
```

b) Install dependencies.

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

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

5. To run code quality statistical analysis:
```
pylint src
```

## Documentation
### Project best practices
The team has chosen to use the cookie cutter template for [data science](https://drivendata.github.io/cookiecutter-data-science/) to ensure correct and reproducible code. Moreover, using a popular file structure for such project will make understanding and finding the code easier for people not involved in this particular project. Except for refactoring the code to this template, the team has also reduced all code to include only its main functionality. All exploratory code has been moved into a separate notebook in the notebooks folder.

### Pipeline Management
Given the relatively simple nature of the model outputs, i.e. it uses two classes, the team has chosen to report the classification scores (multiple metrics such as precision, recall and f1-score) and also include the confusion matrix. These metrics form an all encompassing overview of the model performance, and allows for critical comparison between different models.

### Code Quality
TODO: The project implements different ways to display code quality information, considers multiple linters, critically analyses linter rules, and proposes new missing ML rules.
