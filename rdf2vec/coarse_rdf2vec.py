from argparse import ArgumentParser
from coarsening import Coarsen
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec as W2V
from kg import KG
from pathlib import Path
from sklearn.model_selection import train_test_split
from random import choice
from random import shuffle
from utils import *

if __name__ == '__main__':

    #Interpret command line arguments
    argument_parser = ArgumentParser(description='Perform rdf2vec algorithm on base and coarse graphs')

    argument_parser.add_argument('-d', '--dataset', help='Name of dataset (default = mutag)', default = 'mutag', type=str)
    argument_parser.add_argument('-a', '--alpha', help='Float value fo alpha hyperparameter(default = 0.5)', default=0.5, type=float)
    argument_parser.add_argument('-e', '--eta', help='Int value fo eta hyperparameter(default = 10)', default=10, type=int)
    argument_parser.add_argument('-b', '--bepochs', help='Integer value of number of base graph training epochs (default = 35)', default=35, type=int)    
    argument_parser.add_argument('-c', '--cepochs', help='Integer value of number of coarse graph training epochs (default = 35)', default=35, type=int)    
    argument_parser.add_argument('-f', '--fepochs', help='Integer value of number of fine tuning training epochs (default = 0)', default=0, type=int)
    argument_parser.add_argument('-v', '--validation', help='Flag indicating to test on vaildation set', action='store_true')
    arguments = argument_parser.parse_args()

    dataset = arguments.dataset
    alpha = arguments.alpha
    eta = arguments.eta
    bepochs = arguments.bepochs
    cepochs = arguments.cepochs
    fepochs = arguments.fepochs
    validation = arguments.validation

    print('\n============ RDF2VEC ============')
    print('Dataset:', dataset)
    print('Alpha:', alpha)
    print('Eta:', eta)
    print('Base training epochs:', bepochs)
    print('Coarse training epochs:', cepochs)
    print('Fine tuning training epochs:', fepochs)
    print('Use validation set:', validation)
    print('================================')

    training_entities, training_labels, testing_entities, testing_labels = read_key_entities(dataset)

    if validation:
        training_entities, training_labels, testing_entities, testing_labels = train_test_split(training_entities, training_labels, test_size = 0.2)
        training_entities, training_labels, testing_entities, testing_labels = list(training_entities), list(training_labels), list(testing_entities), list(testing_labels)

    print("\nLoading base knowledge graph...", end = '')
    base_graph = KG()
    base_graph.read_dataset(get_data_path(dataset))
    base_graph.generate_symbolic('base')
    base_entities = base_graph.get_entities()
    uri2symbolic = base_graph.get_uri2symbolic()
    str2uri = base_graph.get_str2uri()
    print("knowledge graph loaded!")

    #Coarsen base graph
    print('Coarsening base graph...', end = '')
    coarse_triples, coarse_entities, coarse_predicates, parents = Coarsen(base_graph.get_symbolic_triples(), base_graph.get_symbolic_entities(), alpha)
    print('coarsening complete!')

    #Hyperparameters of rdf2vec, constants in our experiments
    embedding_dimension = 200
    walk_length = 4

    print('\nGenerating random walks...', end='')
    #Generate random walks on base knowledge graph
    base_corpus = base_graph.generate_walks(training_entities + testing_entities, walk_length)

    with Path('coarse_graph.ttl').open('w', encoding="utf-8") as output_file:
        for triple in coarse_triples:
            output_file.write("<" + str(triple[0]) + "> <" + str(triple[1]) + "> <" + str(triple[2]) + ">.\n")
    
    coarse_graph = KG()
    coarse_graph.read_dataset('coarse_graph.ttl', format = 'ttl')
    coarse_training_entities = {parents[uri2symbolic[entity]] for entity in training_entities}
    coarse_testing_entities = {parents[uri2symbolic[entity]] for entity in testing_entities}
    coarse_corpus = coarse_graph.generate_walks(coarse_training_entities |coarse_testing_entities, walk_length)
    
    base_corpus = base_corpus[:len(coarse_corpus)]
    shuffle(base_corpus)
    shuffle(coarse_corpus)
    print('random walks complete!')

    print("\nNumber of base triples:", len(base_graph.get_triples()), '\t', "Number of base entities:", len(base_entities), '\t', "Number of base predicates:", len(base_graph.get_predicates()), '\t', 'Base corpus size', len(base_corpus))
    print('Number of coarse triples:', len(coarse_triples), '\t', 'Number of coarse entities:', len(coarse_entities), '\t', 'Number of coarse predicates:', len(coarse_predicates), '\t', 'Coarse corpus size', len(coarse_corpus))

    #Initialize base rdf2vec model
    base_rdf2vec = W2V(size=embedding_dimension, min_count=1, window=5)
    base_rdf2vec.build_vocab(base_corpus)
    total_examples = base_rdf2vec.corpus_count

    #Train model
    print('\nTraining on base graph...', end = ' ')
    for epoch in range(bepochs):
        base_rdf2vec.train(base_corpus, total_examples = total_examples, epochs = 1)
    print('training complete!')

    #Obtain embeddings and calculate accuracy
    training_embeddings = [base_rdf2vec.wv.get_vector(str(entity)) for entity in training_entities]
    testing_embeddings = [base_rdf2vec.wv.get_vector(str(entity)) for entity in testing_entities]
    base_accuracy = calculate_accuracy(training_embeddings, testing_embeddings, training_labels, testing_labels)
    print('Base accuracy', base_accuracy, '\t', 'Base training steps', len(base_corpus) * bepochs)
    
    #Initialize coarse model
    coarse_rdf2vec = W2V(size = embedding_dimension, min_count = 1, window = 5)
    coarse_rdf2vec.build_vocab(coarse_corpus)
    total_examples = coarse_rdf2vec.corpus_count

    #Train coarse model
    print('\nTraining on coarse graph...', end = ' ')
    for epoch in range(cepochs):
        coarse_rdf2vec.train(coarse_corpus, total_examples = total_examples, epochs = 1)
    print('Training complete!')

    #Obtain embeddings and calculate accuracy
    training_embeddings = [coarse_rdf2vec.wv.get_vector(str(parents[uri2symbolic[entity]])) for entity in training_entities]
    testing_embeddings = [coarse_rdf2vec.wv.get_vector(str(parents[uri2symbolic[entity]])) for entity in testing_entities]
    coarse_accuracy = calculate_accuracy(training_embeddings, testing_embeddings, training_labels, testing_labels)
    print('Coarse accuracy', coarse_accuracy, '\t', 'Coarse training steps', len(coarse_corpus) * cepochs)
    
    #Only fine tune if fepochs > 0
    if fepochs > 0:

        #Save coarse rdf2vec model... needed reverse mapping
        with Path('coarse_rdf2vec').open('w', encoding="utf-8") as output_file:
            output_file.write(str(len(base_entities)) + " " + str(embedding_dimension) + "\n")
            for entity in base_entities:
                if str(parents[uri2symbolic[entity]]) in coarse_rdf2vec.wv.vocab:
                    output_file.write(str(uri2symbolic[entity]) + " " +  str(" ". join( [str(float(embedding)) for embedding in coarse_rdf2vec.wv.get_vector(str(parents[uri2symbolic[entity]]))] )) + "\n")
                else:
                    output_file.write(str(uri2symbolic[entity]) + " " +  str(" ". join( [str(float(embedding)) for embedding in coarse_rdf2vec.wv.get_vector(choice(list(coarse_rdf2vec.wv.vocab)))] )) + "\n")
        
        #Reverse map coarse embeddings back down to the base level
        print('\nFine tuning...', end = ' ')
        fine_tune_corpus = [[str(uri2symbolic[str2uri[entity]]) for entity in walk] for walk in base_corpus]
        fine_tune_rdf2vec = W2V(size = embedding_dimension, min_count = 1, window = 5)
        fine_tune_rdf2vec.build_vocab(fine_tune_corpus)
        total_examples = fine_tune_rdf2vec.corpus_count
        temp = KeyedVectors.load_word2vec_format("coarse_rdf2vec", binary=False)
        fine_tune_rdf2vec.build_vocab([list(temp.vocab.keys())], update=True)
        fine_tune_rdf2vec.intersect_word2vec_format("coarse_rdf2vec", binary=False, lockf=1.0)

        #Fine tune coarse embeddings
        for epoch in range(fepochs):
            fine_tune_rdf2vec.train(fine_tune_corpus, total_examples = total_examples, epochs = 1)
        print('Fine tuning complete!')

        #Print fine tuning accuracy
        training_embeddings = [fine_tune_rdf2vec.wv.get_vector(str(uri2symbolic[entity])) for entity in training_entities]
        testing_embeddings = [fine_tune_rdf2vec.wv.get_vector(str(uri2symbolic[entity])) for entity in testing_entities]
        fine_tune_accuracy = calculate_accuracy(training_embeddings, testing_embeddings, training_labels, testing_labels)
        print('Fine tuning accuracy', fine_tune_accuracy, '\t', 'Fine tuning training steps', (len(base_corpus) * bepochs) +  (len(coarse_corpus) * cepochs))
        