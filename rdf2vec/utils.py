from rdflib import URIRef
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def read_key_entities(dataset):

    #Assign correct headers for each dataset
    if dataset == 'mutag':
        label_header = 'label_mutagenic' 
        entity_header = 'bond'
    if dataset == 'aifb':
        label_header = 'label_affiliation' 
        entity_header = 'person'
    if dataset == 'bgs':
        label_header = 'label_lithogenesis' 
        entity_header = 'rock'
    if dataset == 'am':
        label_header = 'label_category' 
        entity_header = 'proxy'

    #Read training entities
    training_data = read_csv('data/' + dataset + '/trainingSet.tsv', sep='\t', header=0)
    training_entities = [URIRef(entity) for entity in training_data[entity_header]]
    training_labels = list(training_data[label_header])

    #Read testing entities
    testing_data = read_csv('data/' + dataset + '/testSet.tsv', sep='\t', header=0)
    testing_entities = [URIRef(entity) for entity in testing_data[entity_header]]
    testing_labels = list(testing_data[label_header])

    return training_entities, training_labels, testing_entities, testing_labels

def get_data_path(dataset):

    if dataset == 'mutag':
        return 'data/mutag/mutag_stripped.nt'
    if dataset == 'aifb':
        return 'data/aifb/aifb_stripped.nt'
    if dataset == 'bgs':
        return 'data/bgs/bgs_stripped.nt'
    if dataset == 'am':
        return 'data/am/am_stripped.nt'

def calculate_accuracy(training_embeddings, testing_embeddings, training_labels, testing_labels):

    svm = SVC()
    svm.fit(training_embeddings, training_labels)
    accuracy = accuracy_score(testing_labels, svm.predict(testing_embeddings))
    return accuracy