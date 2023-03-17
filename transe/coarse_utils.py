
from numpy import array
from rdflib import Graph
from rdflib import URIRef
from pandas import read_csv
from pathlib import Path

def interpret_dataset(dataset):
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

def read_dataset(dataset):

    graph = Graph()
    graph.parse('data/' + dataset + '/' + dataset + '_stripped.nt',  format="nt")

    triples = set()
    entities = set()
    predicates = set()

    for s, p, o in graph:
        triples.add((s, p, o))
        entities.add(s)
        entities.add(o)
        predicates.add(p)

    return triples, entities, predicates

def write_dataset(dataset, triples, entities, predicates, coarse = False):
    
    entity_counter = 0
    predicate_counter = 0

    entity2id = {}
    id2entity = {}
    predicate2id = {}
    id2predicate = {}
    str2id = {}

    for triple in triples:
        s, p, o = triple[0], triple[1], triple[2]
        if s not in entity2id:
            entity2id[s] = entity_counter
            id2entity[entity_counter] = s
            str2id[str(s)] = entity_counter
            entity_counter += 1

        if p not in predicate2id:
            predicate2id[p] = predicate_counter
            id2predicate[predicate_counter] = p
            str2id[str(p)] = predicate_counter
            predicate_counter += 1

        if o not in entity2id:
            entity2id[o] = entity_counter
            id2entity[entity_counter] = o
            str2id[str(o)] = entity_counter
            entity_counter += 1

    assert len(entities) == len(entity2id) == len(id2entity)
    assert len(predicates) == len(predicate2id) == len(id2predicate)

    if coarse:
        path = 'data/' + dataset + '/coarse/train2id.txt'
    else:
        path = 'data/' + dataset + '/train2id.txt'
    with Path(path).open('w', encoding="utf-8") as output_file:
        output_file.write(str(len(triples)) + "\n")
        for triple in triples:
            s, p, o = triple[0], triple[1], triple[2]
            output_file.write(str(entity2id[s]) + " " + str(entity2id[o]) + " " + str(predicate2id[p]) +  "\n")

    if coarse:
        path = 'data/' + dataset + '/coarse/entity2id.txt'
    else:
        path = 'data/' + dataset + '/entity2id.txt'
    with Path(path).open('w', encoding="utf-8") as output_file:
        output_file.write(str(len(entities)) + "\n")
        for entity_id in range(len(entities)):
            output_file.write(str("pass") + "\t" + str(entity_id) + "\n")

    if coarse:
        path = 'data/' + dataset + '/coarse/relation2id.txt'
    else:
        path = 'data/' + dataset + '/relation2id.txt'
    with Path(path).open('w', encoding="utf-8") as output_file:
        output_file.write(str(len(predicates)) + "\n")
        for predicate_id in range(len(predicates)):
            output_file.write(str(id2predicate[predicate_id]) + "\t" + str(predicate_id) + "\n")

    return entity2id

def get_learned_embeddings(transe):

    for x in transe.named_children():
        if x[0] == 'ent_embeddings':
            learned_embeddings = x[1].cpu().weight.data.detach().numpy()

    return learned_embeddings

def get_embeddings_for_entities(entities, learned_embeddings, entity2id):

    embeddings = []
    for entity in entities:
        embeddings.append(learned_embeddings[entity2id[entity]])

    return array(embeddings)
