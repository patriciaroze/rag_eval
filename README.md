## Introduction

This repo is a demo of the ```giskard``` library on a RAG model and a classification model.
Link to the Giskard doc : https://docs.giskard.ai/en/latest/index.html

## Step by Step guide
### Requirements
All package dependencies are specified in the poetry.lock files. However, if you wish to run the Giskard Hub, you should have Docker installed.

### Installations
To install the repository dependencies run :
```shell script
poetry install
```

### Create a .env file

Add a ```.env``` file and add an Open AI API key (to run the RAG model). 
You will later on also need to add your Giskard API key to the same file.

```shell script
touch .env
nano .env
```
Write the following in the nano editor (replacing sk-rh**** with your key) and save your changes.

```
OPENAI_API_KEY = sk-rh*****************
```

### Generate local report

Two models can be evaluated in this repository.

In order to evaluate the RAG model :
```shell script
python -m rag_eval --model rag
```

In order to evaluate the classification model :
```shell script
python -m rag_eval --model titanic
```

You can specify a custom path for the HTML report by using the ```--report_path``` argument :
```shell script
python -m rag_eval --model rag --report_path [path_to_html]
```

An example report for the model is committed to the repo at ```scan_report_gpt.html```

### Generate tests on the Hub

Additionnally, you can refer to the ```giskard``` documentation for instructions : https://docs.giskard.ai/en/latest/giskard_hub/installation_hub/install_on_prem/index.html

#### Launch Docker
This repo runs the Giskard Hub in a local docker image.
You should make sure Docker is running before using the Hub.

### Start the Hub

```
giskard hub start
```
You will be able to open the ```giskard``` hub at ```http://localhost:19000/```

Follow instructions on the Hub and generate an API key, add it to your .env file like so:

```
GISKARDHUB_API_KEY = gsk-*******************
```

### Start the ML worker
```
giskard worker start -k -u http://localhost:19000/
```

### Upload objects to the Hub

Only the classification model is supported to be uploaded to the Hub in this repository.

Upload assets to the server by runing :

```
python -m rag_eval --model titanic --to_hub
```
