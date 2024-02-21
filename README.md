# Named Entity Recognition 

This is a Github repository to train Named Entity Recognition models. 

## Usage

Poetry is needed to run this project. 
To change the dataset and model among other things, go to python/ner/config.py 
Steps to follow
1. Clone the project
2. Go to the python folder and do poetry lock -> poetry install
3. Run cli.py train <output_directory>

Alternatively you can build a docker container from the provided dockerfile and run the code on the container.

## Model

DeBERTa v3 large model is used in this project. 

## Dataset

The NER model is trained on the few-nerd dataset from DFKI. The repository could be used with any NER dataset with small changes.
The Token labels are Person, Organisation, Location, Building, Event, Product, Art & Misc.

