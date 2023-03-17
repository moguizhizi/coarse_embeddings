from numpy import random
from rdflib import Graph
from rdflib import Namespace
from uuid import uuid4

class KG:

    def __init__(self):

        self.graph = Graph()
        self.triples = set()
        self.entities = set()
        self.predicates = set()

        self.neighbours = {}

        self.symbolic_triples = set()
        self.symbolic_entities = set()
        self.symbolic_predicates = set()
        self.uri2symbolic = {}
        self.symbolic2uri = {}
        self.str2uri = {}


    #Read in the dataset in .nt format
    def read_dataset(self, path, format = 'nt'):

        self.graph.parse(path, format=format)

        for s, p, o in self.graph:
            self.triples.add((s, p, o))
            self.entities.update((s,o))
            self.predicates.add(p)

            if s not in self.neighbours:
                self.neighbours[s] = set()
            self.neighbours[s].add((s,p,o))

            if (s,p,o) not in self.neighbours:
                self.neighbours[(s,p,o)] = o
            #self.neighbours[(s,p,o)].add(o)

            if o not in self.neighbours:
                self.neighbours[o] = set()
    
    #Transform graph to use symbolic identifiers (helps avoid format errors)
    def generate_symbolic(self, namespace):

        namespace = Namespace('file:///' + namespace + '_namespace/')

        for triple in self.triples:
            s, p, o = triple[0], triple[1], triple[2]
            if s not in self.uri2symbolic:
                symbolic = namespace[str(uuid4())]
                self.uri2symbolic[s] = symbolic
                self.symbolic2uri[symbolic] = s
                self.str2uri[str(s)] = s
            if p not in self.uri2symbolic:
                symbolic = namespace[str(uuid4())]
                self.uri2symbolic[p] = symbolic
                self.symbolic2uri[symbolic] = p
                self.str2uri[str(p)] = p
            if o not in self.uri2symbolic:
                symbolic = namespace[str(uuid4())]
                self.uri2symbolic[o] = symbolic
                self.symbolic2uri[symbolic] = o
                self.str2uri[str(o)] = o

            self.symbolic_triples.add((self.uri2symbolic[s], self.uri2symbolic[p], self.uri2symbolic[o]))
            self.symbolic_entities.update((self.uri2symbolic[s], self.uri2symbolic[o]))
            self.symbolic_predicates.add(self.uri2symbolic[p])
    
        assert len(self.uri2symbolic) == len(self.symbolic2uri)
        assert len(self.symbolic_triples) == len(self.triples)
        assert len(self.symbolic_entities) == len(self.entities)
        assert len(self.symbolic_predicates) == len(self.predicates)

    #Function for generating random walks on knowledge graph. Inspired by https://github.com/IBCNServices/pyRDF2Vec
    def generate_walks(self, walk_entities, walk_length, max_walks = 200):
        walks = set()
        for entity in walk_entities:
            entity_walks = []
            self.visited_entities = set()
            
            while len(entity_walks) < max_walks:
                current_walk = (entity,)
                while len(current_walk) < walk_length * 2:
                    
                    predicates = self.neighbours[current_walk[-1]]
                    hops =[]
                    for pred in predicates:
                        hops.append((pred, self.neighbours[pred]))

                    unvisited_neighbours = [hop for hop in hops if (hop, len(current_walk)) not in self.visited_entities]
                    if len(unvisited_neighbours) == 0:
                        if len(current_walk) > 2:
                            self.visited_entities.add(((current_walk[-2], current_walk[-1]), len(current_walk) - 2))
                        break
                    hop = random.choice(range(len(unvisited_neighbours)))
                    hop = unvisited_neighbours[hop]
                    if len(current_walk) == ((walk_length * 2) - 1):
                        self.visited_entities.add((hop, len(current_walk)))
                    
                    current_walk = current_walk + (hop[0], hop[1])

                entity_walks.append(current_walk)

            entity_walks = list(set(entity_walks))
            walks.update(entity_walks)

        walks_str = []
        for walk in list(walks):
            walk_str = []
            for hop in walk:
                if type(hop) == tuple:
                    hop = hop[1]
                walk_str.append(str(hop))
            walks_str.append(walk_str)

        return walks_str

    #Getters
    def get_triples(self):
        return self.triples
    
    def get_entities(self):
        return self.entities

    def get_predicates(self):
        return self.predicates

    def get_symbolic_triples(self):
        return self.symbolic_triples
    
    def get_symbolic_entities(self):
        return self.symbolic_entities

    def get_symbolic_predicates(self):
        return self.symbolic_predicates

    def get_uri2symbolic(self):
        return self.uri2symbolic

    def get_str2uri(self):
        return self.str2uri
