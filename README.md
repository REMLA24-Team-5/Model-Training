# REMLA-team5
Simple ML application containing the progress made in the REMLA at the TU Delft

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

e) After implementing changes, run the pipeline and push changes, if any, to the remote

```
$ dvc repro
$ dvc push
```

f) To run experiments and see metrics, do the following commands

* Show metrics after dvc repro run
```
$ dvc repro
$ dvc metrics show
```
* Run an experiment:
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