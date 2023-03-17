from argparse import ArgumentParser
from coarsening import *
from coarse_utils import *
from numpy import array
from openke.config import Trainer
from openke.module.loss import MarginLoss
from openke.module.model import TransE
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def initialize_model(coarse):

    if coarse:
        path = './data/' + dataset +'/coarse/'
    else:
        path = './data/' + dataset +'/'

    train_dataloader = TrainDataLoader(in_path = path, nbatches = 100, threads = 8, sampling_mode = "normal", bern_flag = 1, filter_flag = 1, neg_ent = 25, neg_rel = 0)

    #Define the model
    transe = TransE(ent_tot = train_dataloader.get_ent_tot(), rel_tot = train_dataloader.get_rel_tot(), dim = dimension, p_norm = 1, norm_flag = True)

    #Define the loss function
    model = NegativeSampling(model = transe, loss = MarginLoss(margin = 5.0), batch_size = train_dataloader.get_batch_size())

    return train_dataloader, transe, model

if __name__ == '__main__':

    #Interpret command line arguments
    argument_parser = ArgumentParser(description='Perform rdf2vec algorithm on base and coarse graphs')

    argument_parser.add_argument('-d', '--dataset', help='Name of dataset (default = mutag)', default = 'mutag', type=str)
    argument_parser.add_argument('-a', '--alpha', help='Float value fo alpha hyperparameter(default = 0.5)', default = 0.5, type=float)
    argument_parser.add_argument('-e', '--eta', help='Int value fo eta hyperparameter(default = 10)', default=10, type=int)
    argument_parser.add_argument('-b', '--bepochs', help='Integer value of number of base graph training epochs (default = 5)', default = 5, type=int)    
    argument_parser.add_argument('-c', '--cepochs', help='Integer value of number of coarse graph training epochs (default = 5)', default = 5, type=int)    
    argument_parser.add_argument('-v', '--validation', help='Flag indicating to test on vaildation set', action='store_true')
    arguments = argument_parser.parse_args()

    dataset = arguments.dataset
    alpha = arguments.alpha
    eta = arguments.eta
    bepochs = arguments.bepochs
    cepochs = arguments.cepochs
    validation = arguments.validation

    print('\n============ RDF2VEC ============')
    print('Dataset:', dataset)
    print('Alpha:', alpha)
    print('Eta:', eta)
    print('Base training epochs:', bepochs)
    print('Coarse training epochs:', cepochs)
    print('Use validation set:', validation)
    print('================================')

    #Embedding dimension
    dimension = 200

    training_entities, training_labels, testing_entities, testing_labels = interpret_dataset(dataset)

    if validation:
        training_entities, training_labels, testing_entities, testing_labels = train_test_split(training_entities, training_labels, test_size = 0.2)
        training_entities, training_labels, testing_entities, testing_labels = list(training_entities), list(training_labels), list(testing_entities), list(testing_labels)

    base_triples, base_entities, base_predicates = read_dataset(dataset)
    base_entity2id = write_dataset(dataset, base_triples, base_entities, base_predicates, False)

    #Train the model
    base_train_dataloader, base_transe, base_model = initialize_model(False)
    trainer = Trainer(model = base_model, data_loader = base_train_dataloader, train_times = 0, alpha = 1.0, use_gpu = True)
    trainer.run()

    for epoch in range(bepochs):
        for data in trainer.data_loader:
            assert data['batch_y'].shape[0] == trainer.data_loader.batch_seq_size
            loss = trainer.train_one_step(data)
    
    base_transe.save_checkpoint('base_transe.ckpt')
    base_transe.load_checkpoint('base_transe.ckpt')
    
    learned_embeddings = get_learned_embeddings(base_transe)

    training_embeddings = get_embeddings_for_entities(training_entities, learned_embeddings, base_entity2id)
    testing_embeddings = get_embeddings_for_entities(testing_entities, learned_embeddings, base_entity2id)

    #Obtain embedding accuracy
    classifier = SVC()
    classifier.fit(training_embeddings, training_labels)
    base_accuracy = accuracy_score(testing_labels, classifier.predict(testing_embeddings))
    base_training_steps = data['batch_h'].shape[0] * len(list(trainer.data_loader)) * bepochs

    coarse_triples, coarse_entities, coarse_predicates, parents = Coarsen(base_triples, base_entities, alpha)
    coarse_entity2id = write_dataset(dataset, coarse_triples, coarse_entities, coarse_predicates, True)

    coarse_train_dataloader, coarse_transe, coarse_model = initialize_model(True)

    trainer = Trainer(model = coarse_model, data_loader = coarse_train_dataloader, train_times = 0, alpha = 1.0, use_gpu = True)
    trainer.run()

    for epoch in range(cepochs):
        for data in trainer.data_loader:
            assert data['batch_y'].shape[0] == trainer.data_loader.batch_seq_size
            loss = trainer.train_one_step(data)
            
    coarse_transe.save_checkpoint('coarse_transe.ckpt')
    coarse_transe.load_checkpoint('coarse_transe.ckpt')

    learned_embeddings = get_learned_embeddings(coarse_transe)

    training_embeddings = []
    for entity in training_entities:
        training_embeddings.append(learned_embeddings[coarse_entity2id[parents[entity]]])
    training_embeddings =  array(training_embeddings)

    testing_embeddings = []
    for entity in testing_entities:
        testing_embeddings.append(learned_embeddings[coarse_entity2id[parents[entity]]])
    testing_embeddings =  array(testing_embeddings)

    classifier = SVC()
    classifier.fit(training_embeddings, training_labels)
    coarse_accuracy = accuracy_score(testing_labels, classifier.predict(testing_embeddings))
    coarse_training_steps = data['batch_h'].shape[0] * len(list(trainer.data_loader)) * cepochs

    print("\nNumber of base triples:", len(base_triples), '\t', "Number of base entities:", len(base_entities), '\t', "Number of base predicates:", len(base_predicates))
    print('Number of coarse triples:', len(coarse_triples), '\t', 'Number of coarse entities:', len(coarse_entities), '\t', 'Number of coarse predicates:', len(coarse_predicates))

    print('\nBase accuracy', base_accuracy, '\t', 'Base training steps', base_training_steps)
    print('Coarse accuracy', coarse_accuracy, '\t', 'Coarse training steps', coarse_training_steps)
