import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag

#ulit
def load_train_data(file_name, sample2id):
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')

            feature.append([sample2id[tokens[0]]])
            label.append([int(tokens[1])])

    return feature, label


def load_mapping(mapping_file):

    mapping = {}

    file_handle = open(mapping_file)

    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    
    return mapping

def prepare_train_data(train_file, test_file, sample2id_mapping_file):

    # load mapping files
    sample2id_mapping = load_mapping(sample2id_mapping_file)

    train_feature, train_label = load_train_data(train_file, sample2id_mapping)
    test_feature, test_label = load_train_data(test_file, sample2id_mapping)

    print('Total number of samples = %d' % len(sample2id_mapping))

    return (torch.Tensor(train_feature), torch.LongTensor(train_label), torch.Tensor(test_feature), torch.LongTensor(test_label)), sample2id_mapping

def load_ontology(file_name, gene2id_mapping):

    dG = nx.DiGraph()
    term_direct_gene_map = {}
    term_size_map = {}

    file_handle = open(file_name)

    gene_set = set()

    for line in file_handle:

        line = line.rstrip().split()
        
        if line[2] == 'default':
            dG.add_edge(line[0], line[1])
        else:
            if line[1] not in gene2id_mapping:
                continue

            if line[0] not in term_direct_gene_map:
                term_direct_gene_map[ line[0] ] = set()

            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

            gene_set.add(line[1])

    file_handle.close()

    print('There are %d genes' % len(gene_set))

    for term in dG.nodes():
        
        term_gene_set = set()

        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]

        deslist = nxadag.descendants(dG, term)

        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]

        # jisoo
        if len(term_gene_set) == 0:
            print('There is empty terms, please delete term: %s' % term)
            sys.exit(1)
        else:
            term_size_map[term] = len(term_gene_set)

    #leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
    leaves = [n for n,d in dG.in_degree() if d==0]
    #leaves = [n for n,d in dG.in_degree() if d==0]

    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    print('There are %d roots: %s' % (len(leaves), leaves[0]))
    print('There are %d terms' % len(dG.nodes()))
    print('There are %d connected components' % len(connected_subG_list))

    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print('There are more than connected components. Please connect them.')
        sys.exit(1)

    return dG, leaves[0], term_size_map, term_direct_gene_map

def build_input_vector(row_data, num_col, original_features):

    cuda_features = torch.zeros(len(row_data), num_col)

    for i in range(len(row_data)):
        data_ind = row_data[i]
        cuda_features.data[i] = original_features.data[data_ind]
   
    return cuda_features


