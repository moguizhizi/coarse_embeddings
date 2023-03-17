import rdflib
import time
import random

def Coarsen(base_triples, base_entities = None, threshold = 0.5, eta = 10):

    #If base entities not given, obtain them
    if base_entities == None:
        base_entities = set()
        for triple in base_triples:
            base_entities.update([triple[0], triple[2]])
    
    #Define rdf namespace for the coarse graph
    coarse_namespace = rdflib.Namespace("file:///coarse_namespace/")

    neighbours = {}
    tags = {entity: set() for entity in base_entities}
    reverse_tags = {}
    neighbours = {entity: set() for entity in base_entities}

    #Obtain neighbourhood and tag information
    for edge in base_triples:

        subject, predicate, object_ = edge[0], edge[1], edge[2]

        neighbours[subject].update([subject, object_])
        neighbours[object_].update([subject, object_])

        tags[subject].add((predicate, object_))
        tags[object_].add((subject, predicate))

        if (predicate, object_) not in reverse_tags:
            reverse_tags[(predicate, object_)] = []
        reverse_tags[(predicate, object_)].append(subject)

        if (subject, predicate) not in reverse_tags:
            reverse_tags[(subject, predicate)] = []
        reverse_tags[(subject, predicate)].append(object_)

    coarse_triples = set()
    coarse_entities = set()
    coarse_predicates = set()
    parents = {entity:None for entity in base_entities}
    counter = 1

    #Perform second order collapsing
    for entity in base_entities:

        if parents[entity] != None:
            continue
        for iteration in range(eta):
            first_order_tag = random.sample(tags[entity], 1)[0]
            second_order_neighbour = random.sample(reverse_tags[first_order_tag], 1)[0]

            if parents[second_order_neighbour] == None and second_order_neighbour != entity:

                jaccard = float(len(tags[entity].intersection(tags[second_order_neighbour]))) / float(len(tags[entity].union(tags[second_order_neighbour])))
                
                if jaccard >= threshold:
                    if parents[entity] == None:
                        new_cluster = coarse_namespace[str(counter)]
                        counter += 1
                        assert parents[entity] == None
                        parents[entity] = new_cluster
                
                    assert parents[second_order_neighbour] == None
                    parents[second_order_neighbour] = parents[entity]

    #Perform first order collapsing
    for entity in base_entities:
        if parents[entity] == None:
            for iteration in range(eta):

                first_order_neighbour = random.sample(neighbours[entity], 1)[0]
                if entity == first_order_neighbour:
                    continue

                if neighbours[entity].issubset(neighbours[first_order_neighbour]):

                    entity_tags = {tag for tag in tags[entity] if first_order_neighbour not in tag}
                    first_order_neighbour_tags = {tag for tag in tags[first_order_neighbour] if entity not in tag}
                    
                    if entity_tags.issubset(first_order_neighbour_tags):
                        if parents[entity] == None:
                            if parents[first_order_neighbour] == None:
                                new_cluster = coarse_namespace[str(counter)]
                                counter += 1
                                assert parents[entity] == None
                                parents[entity] = new_cluster
                                assert parents[first_order_neighbour] == None
                                parents[first_order_neighbour] = new_cluster
                            else:
                                assert parents[entity] == None
                                parents[entity] = parents[first_order_neighbour]
                         
        if parents[entity] == None:
            new_cluster = coarse_namespace[str(counter)]
            counter += 1
            assert parents[entity] == None
            parents[entity] = new_cluster

    #Generate coarse graph
    for triple in base_triples:
        subject, predicate, object_ = triple[0], triple[1], triple[2]
        new_subject, new_object_ = subject, object_
        if subject in parents:
            assert parents[subject] != None
            new_subject = parents[subject]
        if object_ in parents:
            assert parents[object_] != None
            new_object_ = parents[object_]
        if new_subject != new_object_:
            coarse_triples.add((new_subject, predicate, new_object_))
            coarse_entities.update([new_subject, new_object_])
            coarse_predicates.add(predicate)
    
    return (coarse_triples, coarse_entities, coarse_predicates, parents)