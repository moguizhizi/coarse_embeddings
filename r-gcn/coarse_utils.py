from pandas import read_csv
from rdflib import Graph
from rdflib import Namespace
from rgcn.utils import *
from data_utils import *
from uuid import uuid4

def reverse_map(dataset, predictions, parents, uri2symbolic, str2uri, label2id, y_test):

    test_entities = np.load('data/' + dataset + '/test_names.npy')
    test_idx = np.load('data/' + dataset + '/test_idx.npy')
    test_labels = np.load('data/' + dataset + '/labels.npz')['indices']
    indptr = np.load('data/' + dataset + '/labels.npz')['indptr']

    number_of_categories = np.unique(test_labels).shape[0]

    y_test_transfer = np.zeros((indptr.shape[0], number_of_categories))
    idx_test_transfer = []
    for test_entity in range(test_entities.shape[0]):
        y_test_transfer[test_idx[test_entity], test_labels[indptr[test_idx[test_entity]]] % number_of_categories] = 1
        idx_test_transfer.append(test_idx[test_entity])
    idx_test_transfer = np.array(idx_test_transfer)

    mapped_predictions = []
    for test_entity in range(test_entities.shape[0]):
        entity_parent = parents[uri2symbolic[str2uri[str(test_entities[test_entity])]]]
        mapped_predictions.append(predictions[label2id[str(entity_parent)]])

    mapped_predictions = np.array(mapped_predictions)
    coarse_test_labels = y_test[idx_test_transfer]

    return mapped_predictions, coarse_test_labels

def read_graph(dataset):

    graph = Graph()    
    if dataset == 'mutag':
        graph.parse('data/mutag/mutag_stripped.nt',  format="nt")
        label_header = 'label_mutagenic' 
        entity_header = 'bond'
    if dataset == 'aifb':
        graph.parse('data/aifb/aifb_stripped.nt',  format="nt")
        label_header = 'label_affiliation' 
        entity_header = 'person'
    if dataset == 'bgs':
        graph.parse('data/bgs/bgs_stripped.nt',  format="nt")
        label_header = 'label_lithogenesis' 
        entity_header = 'rock'
    if dataset == 'am':
        graph.parse('data/am/am_stripped.nt',  format="nt")
        label_header = 'label_cateogory' 
        entity_header = 'proxy'

    return graph, label_header, entity_header

def symbolize_graph(graph):
    symbolic_namespace = Namespace("file:///symbolic_namespace/")
    uri2symbolic = {}
    symbolic2uri = {}
    str2uri = {}
    
    symbolic_triples = set()
    symbolic_entities = set()
    symbolic_predicates = set()

    for s, p, o in graph:
        if s not in uri2symbolic:
            symbolic = symbolic_namespace[str(uuid4())]
            uri2symbolic[s] = symbolic
            str2uri[s.encode('utf-8', 'ignore').decode('utf-8')] = s
            symbolic2uri[symbolic] = s
        if p not in uri2symbolic:
            symbolic = symbolic_namespace[str(uuid4())]
            uri2symbolic[p] = symbolic
            symbolic2uri[symbolic] = p
        if o not in uri2symbolic:
            symbolic = symbolic_namespace[str(uuid4())]
            uri2symbolic[o] = symbolic
            str2uri[o.encode('utf-8', 'ignore').decode('utf-8')] = o
            symbolic2uri[symbolic] = o

        symbolic_triples.add((uri2symbolic[s], uri2symbolic[p], uri2symbolic[o]))
        symbolic_entities.update([uri2symbolic[s], uri2symbolic[o]])
        symbolic_predicates.add(uri2symbolic[p])

    return symbolic_triples, symbolic_entities, symbolic_predicates, uri2symbolic, symbolic2uri, str2uri

#Create datasets so they may be used by R-GCN
def create_datasets(dataset, output_file, label_header, entity_header, parents, uri2symbolic, symbolic2uri, str2uri):
    
    input_file = read_csv('data/' + dataset + '/' + output_file + '.tsv', sep='\t', encoding='utf-8')
    
    entities = list(input_file[entity_header])
    labels = list(input_file[label_header])
    assert len(entities) == len(labels)

    coarse_labels = {}
    for i in range(len(entities)):
        coarse_labels[parents[uri2symbolic[str2uri[str(entities[i])]]]] = labels[i]

    with open('data/coarse_' +  dataset + '/' + output_file + '.tsv', 'w') as output_file:
        output_file.write(entity_header + "\t" + label_header + "\n")
        for entry in coarse_labels:
            output_file.write(str(entry) + "\t" + str(coarse_labels[entry]) +  "\n")

#R-GCN dataloader, modified to work for coarse graphs 
def load_coarse_graph(graph_file, dataset, label_header, nodes_header, limit=-1):

    labels_df = pd.read_csv('data/' + str(dataset) + '/completeDataset.tsv', sep='\t', encoding='utf-8')
    labels_train_df = pd.read_csv('data/coarse_' + str(dataset) + '/trainingSet.tsv', sep='\t', encoding='utf8')
    labels_test_df = pd.read_csv('data/coarse_' + str(dataset) + '/testSet.tsv', sep='\t', encoding='utf8')

    labels_file = 'data/coarse_' + str(dataset) + '/labels.npz' 
    train_idx_file = 'data/coarse_' + str(dataset) + '/train_idx.npy'
    test_idx_file = 'data/coarse_' + str(dataset) + '/test_idx.npy'
    train_names_file = 'data/coarse_' + str(dataset) + '/train_names.npy'
    test_names_file = 'data/coarse_' + str(dataset) + '/test_names.npy'
    rel_dict_file = 'data/coarse_' + str(dataset) + '/rel_dict.pkl'
    nodes_file = 'data/coarse_' + str(dataset) + '/nodes.pkl'

    with RDFReader(graph_file) as reader:

        relations = reader.relationList()
        subjects = reader.subjectSet()
        objects = reader.objectSet()

        nodes = list(subjects.union(objects))
        adj_shape = (len(nodes), len(nodes))

        relations_dict = {rel: i for i, rel in enumerate(list(relations))}
        nodes_dict = {node: i for i, node in enumerate(nodes)}

        assert len(nodes_dict) < np.iinfo(np.int32).max

        adjacencies = []

        for i, rel in enumerate(
                relations if limit < 0 else relations[:limit]):

            edges = np.empty((reader.freq(rel), 2), dtype=np.int32)

            size = 0
            for j, (s, p, o) in enumerate(reader.triples(relation=rel)):
                if nodes_dict[s] > len(nodes) or nodes_dict[o] > len(nodes):
                    print(s, o, nodes_dict[s], nodes_dict[o])

                edges[j] = np.array([nodes_dict[s], nodes_dict[o]])
                size += 1

            row, col = np.transpose(edges)

            data = np.ones(len(row), dtype=np.int8)

            adj = sp.csr_matrix((data, (row, col)), shape=adj_shape,
                                dtype=np.int8)

            adj_transp = sp.csr_matrix((data, (col, row)), shape=adj_shape,
                                        dtype=np.int8)

            adj_fprepend = os.path.dirname(os.path.realpath(sys.argv[0])) + '/data/coarse_'+ dataset + '/adjacencies_'

            save_sparse_csr(adj_fprepend + '%d.npz' % (i * 2), adj)
            save_sparse_csr(adj_fprepend + '%d.npz' % (i * 2 + 1),
                            adj_transp)

            if limit < 0:
                adjacencies.append(adj)
                adjacencies.append(adj_transp)

    # Reload the adjacency matrices from disk
    if limit > 0:
        adj_files = glob.glob(adj_fprepend + '*.npz')
        adj_files.sort(key=lambda f: int(
            re.search('adjacencies_(.+?).npz', f).group(1)))

        adj_files = adj_files[:limit * 2]
        for i, file in enumerate(adj_files):
            adjacencies.append(load_sparse_csr(file))
            

    nodes_u_dict = {np.unicode(to_unicode(key)): val for key, val in
                        nodes_dict.iteritems()}

    labels_set = set(labels_df[label_header].values.tolist())
    labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}


    labels = sp.lil_matrix((adj_shape[0], len(labels_set)))
    labeled_nodes_idx = []

    train_idx = []
    train_names = []
    for nod, lab in zip(labels_train_df[nodes_header].values,
                        labels_train_df[label_header].values):
        nod = np.unicode(to_unicode(nod))  # type: unicode
        if nod in nodes_u_dict:
            labeled_nodes_idx.append(nodes_u_dict[nod])
            label_idx = labels_dict[lab]
            labels[labeled_nodes_idx[-1], label_idx] = 1
            train_idx.append(nodes_u_dict[nod])
            train_names.append(nod)
        else:
            print(u'Node not in dictionary, skipped: ',
                    nod.encode('utf-8', errors='replace'))

    test_idx = []
    test_names = []
    for nod, lab in zip(labels_test_df[nodes_header].values,
                        labels_test_df[label_header].values):
        nod = np.unicode(to_unicode(nod))
        if nod in nodes_u_dict:
            labeled_nodes_idx.append(nodes_u_dict[nod])
            label_idx = labels_dict[lab]
            labels[labeled_nodes_idx[-1], label_idx] = 1
            test_idx.append(nodes_u_dict[nod])
            test_names.append(nod)
        else:
            print(u'Node not in dictionary, skipped: ',
                    nod.encode('utf-8', errors='replace'))

    labeled_nodes_idx = sorted(labeled_nodes_idx)
    labels = labels.tocsr()

    save_sparse_csr(labels_file, labels)

    np.save(train_idx_file, train_idx)
    np.save(test_idx_file, test_idx)

    np.save(train_names_file, train_names)
    np.save(test_names_file, test_names)

    pkl.dump(relations_dict, open(rel_dict_file, 'wb'))
    pkl.dump(nodes, open(nodes_file, 'wb'))

    features = sp.identity(adj_shape[0], format='csr')

    return adjacencies, features, labels, labeled_nodes_idx, train_idx, test_idx, relations_dict, train_names, test_names, nodes_u_dict

def parse(symbol):
    if symbol.startswith('<'):
        return symbol[1:-1]
    return symbol


def to_unicode(input):
    if isinstance(input, unicode):
        return input
    elif isinstance(input, str):
        return input.decode('utf-8', errors='replace')
    return str(input).decode('utf-8', errors='replace')

#R-GCN data preprocessing code to create pickle file, modified for coarse graphs
def create_pickle(dataset, A, X, y, labeled_nodes_idx, train_idx, test_idx, rel_dict, train_names, test_names):

    NUM_GC_LAYERS = 2  # Number of graph convolutional layers

    rel_list = range(len(A))
    for key, value in rel_dict.iteritems():
        if value * 2 >= len(A):
            continue
        rel_list[value * 2] = key
        rel_list[value * 2 + 1] = key + '_INV'


    num_nodes = A[0].shape[0]
    A.append(sp.identity(A[0].shape[0]).tocsr())  # add identity matrix

    support = len(A)

    # Get level sets (used for memory optimization)
    bfs_generator = bfs_relational(A, labeled_nodes_idx)
    lvls = list()
    lvls.append(set(labeled_nodes_idx))
    lvls.append(set.union(*bfs_generator.next()))

    # Delete unnecessary rows in adjacencies for memory efficiency
    todel = list(set(range(num_nodes)) - set.union(lvls[0], lvls[1]))
    for i in range(len(A)):
        csr_zero_rows(A[i], todel)

    data = {'A': A, 'y': y, 'train_idx': train_idx, 'test_idx': test_idx}

    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

    with open(dirname + '/coarse_' + dataset + '.pickle', 'wb') as f:
        pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

