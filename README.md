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

e) After implementing changes, run the pipeline and push to the remote

```
$ dvc repro
$ dvc push
```