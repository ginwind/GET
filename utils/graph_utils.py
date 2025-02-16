import dgl
import torch
import networkx as nx

biDirecRelation = ["Synonym", "Antonym", "random", "coord", "SYN", "RANDOM", "ANT", "sibl", "false", "COORD"]
relationDes = {
    "Synonym": "A and B have very similar meanings. They may be translations of each other in different languages. This is the synonym relation in WordNet as well. Symmetric.",
    "HasProperty": "A has B as a property; A can be described as B.",
    "Antonym": "A and B are opposites in some relevant way, such as being opposite ends of a scale, or fundamentally similar things with a key difference between them. Counterintuitively, two concepts must be quite similar before people consider them antonyms. This is the antonym relation in WordNet as well. Symmetric.",
    "IsA": "A is a subtype or a specific instance of B; every A is a B. This can include specific instances; the distinction between subtypes and instances is often blurry in language. This is the hyponym relation in WordNet.",
    "PartOf": "A is a part of B. This is the part meronym relation in WordNet.",
    "MadeOf": "A is made of B.",
    "HasA": "B belongs to A, either as an inherent part or due to a social construct of possession. HasA is often the reverse of PartOf.",
    
    "HYPER": "A is a subtype or a specific instance of B; every A is a B. This can include specific instances; the distinction between subtypes and instances is often blurry in language. This is the hyponym relation in WordNet.",
    "RANDOM": "There is no relation between A and B.",
    'SYN': "A and B have very similar meanings. They may be translations of each other in different languages. This is the synonym relation in WordNet as well. Symmetric.",
    "PART_OF": "A is a part of B. This is the part meronym relation in WordNet.",
    "ANT": "A and B are opposites in some relevant way, such as being opposite ends of a scale, or fundamentally similar things with a key difference between them. Counterintuitively, two concepts must be quite similar before people consider them antonyms. This is the antonym relation in WordNet as well. Symmetric.",

    "mero": "A is a part of B. This is the part meronym relation in WordNet.",
    "hypo": "A is a subtype or a specific instance of B; every A is a B. This can include specific instances; the distinction between subtypes and instances is often blurry in language. This is the hyponym relation in WordNet.",
    "false": "There is no relation between A and B.",
    "sibl": "The most general relation. There is some positive relationship between A and B, but ConceptNet can't determine what that relationship is based on the data. Symmetric.",
    
    "COORD": "The most general relation. There is some positive relationship between A and B, but ConceptNet can't determine what that relationship is based on the data. Symmetric.",

    "random": "There is no relation between A and B.",
    "coord": "The most general relation. There is some positive relationship between A and B, but ConceptNet can't determine what that relationship is based on the data. Symmetric.",
    "attri": "A has B as a property; A can be described as B.",
    "hyper": "A is a subtype or a specific instance of B; every A is a B. This can include specific instances; the distinction between subtypes and instances is often blurry in language. This is the hyponym relation in WordNet.",
    "event": "A and B are events, and B happens as a subevent of A."
}

def create_heterogeneous_entity_graph(train_dataloader, words, relations):
    bi_direc_rel_index = [relations.index(each) for each in biDirecRelation if each in relations]

    # original knowledge graph
    G = nx.DiGraph()
    G.add_nodes_from(range(0, len(words)))
    for data in train_dataloader:
        if len(data["labels"].shape) == 1:
            for i in range(0, data["labels"].shape[0]):
                rel = data["labels"][i].item()
                G.add_edge(data["h"][i].item(), data["t"][i].item(), label=rel if isinstance(rel, int) else 0)
                if rel in bi_direc_rel_index:
                    G.add_edge(data["t"][i].item(), data["h"][i].item(), label=rel if isinstance(rel, int) else 0)
        else:
            edge_labels = data["graph_label"].flatten()
            node_starts = data["h"].flatten()
            node_ends = data["t"].flatten()
            for i in range(0, edge_labels.shape[0]):
                rel = int(edge_labels[i].item())
                G.add_edge(int(node_starts[i].item()), int(node_ends[i].item()), label=rel if isinstance(rel, int) else 0)
                if rel in bi_direc_rel_index:
                    G.add_edge(int(node_ends[i].item()), int(node_starts[i].item()), label=rel if isinstance(rel, int) else 0)

    pr = nx.pagerank(G)
    add_connected_edges = False
    if not nx.is_weakly_connected(G):
        add_connected_edges = True
        weakly_connected_components = list(nx.weakly_connected_components(G))
        for i in range(0, len(weakly_connected_components) - 1):
            nodes_a = max({each: pr[each] for each in list(weakly_connected_components[i])}, key=lambda x: pr[x])
            nodes_b = max({each: pr[each] for each in list(weakly_connected_components[i + 1])}, key=lambda x: pr[x])
            if pr[nodes_a] > pr[nodes_b]:
                G.add_edge(nodes_b, nodes_a, label=len(relations))
            else:
                G.add_edge(nodes_a, nodes_b, label=len(relations))
    assert nx.is_weakly_connected(G)
    pr = nx.pagerank(G)
    return dgl.from_networkx(G, edge_attrs=["label"]), add_connected_edges, [pr[i] for i in range(0, len(pr))]

def create_knowledge_embeddings(words, relations, plm, tokenizer, device, inference_batch=1024):
    word_descriptions, relation_descriptions = words, []
    if len(relations) > 1:
        for each in relations:
            relation_descriptions.append(f"The relation between A and B is {each}, which means that {relationDes[each]}")
    else:
        relation_descriptions = relations

    relation_inputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=relation_descriptions,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True
    ).to(device)
    word_embeddings = []
    with torch.no_grad():
        c_idx = 0
        for current_idx in range(inference_batch, len(word_descriptions), inference_batch):
            word_inputs = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=word_descriptions[current_idx - inference_batch: current_idx],
                add_special_tokens=True,
                return_tensors="pt",
                padding=True
            ).to(device)
            word_embeddings.append(plm(**word_inputs, output_hidden_states=True)["hidden_states"][1: ])
            torch.cuda.empty_cache()
            c_idx = current_idx
        word_inputs = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=word_descriptions[c_idx:],
                add_special_tokens=True,
                return_tensors="pt",
                padding=True
        ).to(device)
        word_embeddings.append(plm(**word_inputs, output_hidden_states=True)["hidden_states"][1: ])
        w_e = []
        for i in range(0, len(word_embeddings[0])):
            layers_embeddings = []
            for j in range(0, len(word_embeddings)):
                layers_embeddings.append(torch.mean(word_embeddings[j][i], dim=1))
            w_e.append(torch.concat(layers_embeddings, dim=0))
        relation_embeddings = plm(**relation_inputs, output_hidden_states=True)["hidden_states"][1: ]
    relation_embeddings = [torch.mean(each, dim=1) for each in relation_embeddings]
    # layers x batch x dim
    return w_e, relation_embeddings