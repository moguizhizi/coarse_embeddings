from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from rgcn.layers.graph import GraphConvolution
from rgcn.layers.input_adj import InputAdj
from rgcn.utils import *

from coarse_utils import *
from coarsening import *

import pickle as pkl

import os
import sys
import time
import argparse

def train(A, y, train_idx, test_idx, epochs, validation, lr, l2_, hidden, bases, dropout):

    # Get dataset splits
    y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y, train_idx, test_idx, validation)
    train_mask = sample_mask(idx_train, y.shape[0])

    num_nodes = A[0].shape[0]
    support = len(A)

    # Define empty dummy feature matrix (input is ignored as we set featureless=True)
    # In case features are available, define them here and set featureless=False.
    X = sp.csr_matrix(A[0].shape)
    
    # Normalize adjacency matrices individually
    for i in range(len(A)):
        d = np.array(A[i].sum(1)).flatten()
        d_inv = 1. / d
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        A[i] = D_inv.dot(A[i]).tocsr()

    A_in = [InputAdj(sparse=True) for _ in range(support)]
    X_in = Input(shape=(X.shape[1],), sparse=True)

    # Define model architecture
    H = GraphConvolution(hidden, support, num_bases = bases, featureless = True, activation = 'relu', W_regularizer = l2(l2_))([X_in] + A_in)
    H = Dropout(dropout)(H)
    Y = GraphConvolution(y_train.shape[1], support, num_bases=bases, activation='softmax')([H] + A_in)

    # Compile model
    model = Model(input = [X_in] + A_in, output = Y)
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = lr))

    predictions = None
    
    # Fit
    for epoch in range(1, epochs + 1):

        # Single training iteration
        model.fit([X] + A, y_train, sample_weight=train_mask,
                batch_size=num_nodes, nb_epoch=1, shuffle=False, verbose=0)
        
        if epoch % 1 == 0:

            # Predict on full dataset
            predictions = model.predict([X] + A, batch_size=num_nodes)

    return model, predictions, y_test, idx_test

def prepare_coarse_graph(dataset, alpha, eta):

    #Read base graph from file
    base_graph, label_header, entity_header = read_graph(dataset)

    #Create symbolic representation of base graph. This makes it easier to deal with later on
    symbolic_triples, symbolic_entities, symbolic_predicates, uri2symbolic, symbolic2uri, str2uri = symbolize_graph(base_graph)

    #Coarsen symbolic graph
    print('\nCoarsening base graph...', end = '')
    coarse_triples, coarse_entities, coarse_predicates, parents = Coarsen(symbolic_triples, symbolic_entities, alpha, eta)
    print('coarsening complete!')
    print('Number of coarse triples:', len(coarse_triples), '\t', 'Number of coarse entities:', len(coarse_entities), '\t', 'Number of coarse predicates:', len(coarse_predicates))

    #Prepare coarse directory so it can be used by R-GCN
    create_datasets(dataset, 'completeDataset', label_header, entity_header, parents, uri2symbolic, symbolic2uri, str2uri)
    create_datasets(dataset, 'trainingSet', label_header, entity_header, parents, uri2symbolic, symbolic2uri, str2uri)
    create_datasets(dataset, 'testSet', label_header, entity_header, parents, uri2symbolic, symbolic2uri, str2uri)
    
    #Save coarse graph
    with open('data/coarse_' + dataset + '/coarse_graph.ttl', 'w') as output_file:
        for triple in coarse_triples:
            output_file.write("<" + str(triple[0]) + "> <" + str(triple[1]) + "> <" + str(triple[2]) + ">.\n")

    #Load/preprocess coarse graph to work for R-GCN
    A, X, y, labeled_nodes_idx, train_idx, test_idx, rel_dict, train_names, test_names, label2id = load_coarse_graph('data/coarse_' + dataset + '/coarse_graph.ttl', dataset, label_header, entity_header)
    
    #Create a pickle file of coarse dataste
    create_pickle(dataset, A, X, y, labeled_nodes_idx, train_idx, test_idx, rel_dict, train_names, test_names)
    
    #Load coarse data
    with open(dirname + '/coarse_' + dataset + '.pickle', 'rb') as f:
        data = pkl.load(f)

    coarse_A = data['A']
    coarse_y = data['y']
    coarse_train_idx = data['train_idx']
    coarse_test_idx = data['test_idx']
    
    return coarse_A, coarse_y, coarse_train_idx, coarse_test_idx, parents, label2id, uri2symbolic, symbolic2uri, str2uri

if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser(description='Perform R-GCN algorithm on base and coarse graphs')
    
    #Arguments from original R-GCN
    argument_parser.add_argument("-d", "--dataset", type=str, default="aifb", help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
    argument_parser.add_argument("-hd", "--hidden", type=int, default=16, help="Number hidden units")
    argument_parser.add_argument("-do", "--dropout", type=float, default=0., help="Dropout rate")
    argument_parser.add_argument("-b", "--bases", type=int, default=-1, help="Number of bases used (-1: all)")
    argument_parser.add_argument("-lr", "--learnrate", type=float, default=0.01, help="Learning rate")
    argument_parser.add_argument("-l2", "--l2norm", type=float, default=0., help="L2 normalization of input weights")
    fp = argument_parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')

    #Arguments for coarsening
    argument_parser.add_argument('-a', '--alpha', help='Float value fo alpha hyperparameter(default = 0.5)', default=0.5, type=float)
    argument_parser.add_argument('-e', '--eta', help='Int value fo eta hyperparameter(default = 10)', default=10, type=int)
    argument_parser.add_argument('-be', '--bepochs', help='Integer value of number of base graph training epochs (default = 20)', default=20, type=int)    
    argument_parser.add_argument('-ce', '--cepochs', help='Integer value of number of coarse graph training epochs (default = 20)', default=20, type=int)    
    
    #Read arguments
    args = vars(argument_parser.parse_args())
    dataset = args['dataset']
    bepochs = args['bepochs']
    cepochs = args['cepochs']
    validation = args['validation']
    lr = args['learnrate']
    l2_ = args['l2norm']
    hidden = args['hidden']
    bases = args['bases']
    dropout = args['dropout']
    alpha = args['alpha']
    eta = args['eta']

    print('\n============ R-GCN ============')
    print('Dataset:', dataset)
    print('Alpha:', alpha)
    print('Eta:', eta)
    print('Base training epochs:', bepochs)
    print('Coarse training epochs:', cepochs)
    print('Use validation set:', validation)
    print('================================')

    np.random.seed()

    #Get current directory name
    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

    #Read base graph
    with open(dirname + '/' + dataset + '.pickle', 'rb') as f:
        data = pkl.load(f)

    base_A = data['A']
    base_y = data['y']
    base_train_idx = data['train_idx']
    base_test_idx = data['test_idx']
    
    #Train R-GCN on base graph
    print('\nTraining on base graph...', end = ' ')
    base_model, predictions, y_test, idx_test = train(base_A, base_y, base_train_idx, base_test_idx, bepochs, validation, lr, l2_, hidden, bases, dropout)
    print('training complete!')

    #Calculate base accuracy
    base_accuracy = np.mean(np.equal(np.argmax(y_test[idx_test], 1), np.argmax(predictions[idx_test], 1)))
    base_training_steps = bepochs * base_y.shape[0]
    print('Base accuracy', base_accuracy, '\t', 'Base training steps', base_training_steps)
    
    #Prepare coarse graph
    coarse_A, coarse_y, coarse_train_idx, coarse_test_idx, parents, label2id, uri2symbolic, symbolic2uri, str2uri = prepare_coarse_graph(dataset, alpha, eta)

    #Train coarse model
    coarse_model, coarse_predictions, coarse_y_test, coarse_idx_test = train(coarse_A, coarse_y, coarse_train_idx, coarse_test_idx, cepochs, validation, lr, l2_, hidden, bases, dropout)
    
    #Reverse map entities to base level
    predictions, coarse_test_labels = reverse_map(dataset, coarse_predictions, parents, uri2symbolic, str2uri, label2id, y_test)

    #Calculate coarse accuracy
    coarse_accuracy = np.mean(np.equal(np.argmax(coarse_test_labels, 1), np.argmax(predictions, 1)))
    coarse_training_steps = cepochs * coarse_y.shape[0]
    print('Coarse accuracy', coarse_accuracy, '\t', 'Coarse training steps', coarse_training_steps)