import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from parsvnn_NN import *
import argparse


# model train

def create_term_mask(term_direct_gene_map, gene_dim):

    term_mask_map = {}

    for term, gene_set in term_direct_gene_map.items():

        mask = torch.zeros(len(gene_set), gene_dim)

        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1

        mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))#

        term_mask_map[term] = mask_gpu

    return term_mask_map

def train_model(root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, model_save_folder, train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_final, exp_features, num_class, CUDA_ID):

    '''
    # arguments:
    # 1) root: the root of the hierarchy embedded in one side of the model
    # 2) term_size_map: dictionary mapping the name of subsystem in the hierarchy to the number of genes contained in the subsystem
    # 3) term_direct_gene_map: dictionary mapping each subsystem in the hierarchy to the set of genes directly contained in the subsystem (i.e., children subsystems would not have the genes in the set)
    # 4) dG: the hierarchy loaded as a networkx DiGraph object
    # 5) train_data: torch Tensor object containing training data (features and labels)
    # 6) gene_dim: the size of input vector for the genomic side of neural network (visible neural network) embedding cell features 
    # 7) drug_dim: the size of input vector for the fully-connected neural network embedding drug structure 
    # 8) model_save_folder: the location where the trained model will be saved
    # 9) train_epochs: the maximum number of epochs to run during the training phase
    # 10) batch_size: the number of data points that the model will see at each iteration during training phase (i.e., #training_data_points < #iterations x batch_size)
    # 11) learning_rate: learning rate of the model training
    # 12) num_hiddens_genotype: number of neurons assigned to each subsystem in the hierarchy
    # 13) num_hiddens_drugs: number of neurons assigned to the fully-connected neural network embedding drug structure - one string containing number of neurons at each layer delimited by comma(,) (i.e. for 3 layer of fully-connected neural network containing 100, 50, 20 neurons from bottom - '100,50,20')
    # 14) num_hiddens_final: number of neurons assigned to the fully-connected neural network combining the genomic side with the drug side. Same format as 13).
    # 15) cell_features: a list containing the features of each cell line in tranining data. The index should match with cell2id list.
    # 16) drug_features: a list containing the morgan fingerprint (or other embedding) of each drug in training data. The index should match with drug2id list.
    '''

    # initialization of variables
    best_model = 0
    #best_model = 0
    max_acc = 0

    # dcell neural network
    model = parsvnn_nn(term_size_map, term_direct_gene_map, dG, gene_dim, root, num_hiddens_genotype, num_hiddens_final, num_class, CUDA_ID)

    # separate the whole data into training and test data
    train_feature, train_label, test_feature, test_label = train_data

    # copy labels (observation) to GPU - will be used to 
    train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
    test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))

    # create a torch objects containing input features for cell lines and drugs
    cuda_exp = torch.from_numpy(exp_features)

    # load model to GPU
    model.cuda(CUDA_ID)

    # define optimizer
    # optimize drug NN
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

    optimizer.zero_grad()
    for name, param in model.named_parameters():
        term_name = name.split('_')[0]

        if '_direct_gene_layer.weight' in name:
            #print(name, param.size(), term_mask_map[term_name].size()) 
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
            #param.data = torch.mul(param.data, term_mask_map[term_name])
        else:
            param.data = param.data * 0.1

    # create dataloader for training/test data
    train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
    test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

    for epoch in range(train_epochs):

        #Train
        model.train()
        train_predict = torch.zeros(0,0).cuda(CUDA_ID)
        train_correct = torch.tensor(0.0).cuda(CUDA_ID)
        for i, (inputdata, labels) in enumerate(train_loader):

            cuda_labels = torch.autograd.Variable(torch.flatten(labels)).cuda(CUDA_ID)
            #cuda_labels = torch.autograd.Variable(labels)#.cuda(CUDA_ID)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer

            cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_exp)
            #cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)
            #print(cuda_cell_features.shape)

            # Here term_NN_out_map is a dictionary 
            aux_out_map = model(cuda_cell_features)

            train_correct += (aux_out_map['final'].data.argmax(1) == cuda_labels).type(torch.float).sum().item()

            total_loss = 0
            for name, output in aux_out_map.items():
                loss = nn.CrossEntropyLoss()
                if name == 'final':
                    total_loss += loss(output, cuda_labels)
                else: # change 0.2 to smaller one for big terms
                    total_loss += 0.2 * loss(output, cuda_labels)

            total_loss.backward()

            for name, param in model.named_parameters():
                if '_direct_gene_layer.weight' not in name:
                    continue
                term_name = name.split('_')[0]
                #print(name, param.grad.data.size(), term_mask_map[term_name].size())
                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])
            
            optimizer.step()
            print(i,total_loss)
            
        size = len(train_loader.dataset)
        train_correct /= size

        #Test: random variables in training mode become static
        model.eval()
            

        test_correct = torch.tensor(0.0).cuda(CUDA_ID)
        for i, (inputdata, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_exp)
            #cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)
            cuda_labels = torch.autograd.Variable(torch.flatten(labels)).cuda(CUDA_ID)

            aux_out_map = model(cuda_cell_features)
            
            test_correct += (aux_out_map['final'].data.argmax(1) == cuda_labels).type(torch.float).sum().item()

        size = len(test_loader.dataset)
        test_correct /= size

    
        print("Epoch\t%d\ttrain_corr\t%.6f\ttest_corr\t%.6f\ttotal_loss\t%.6f" % (epoch, torch.flatten(train_correct),torch.flatten(test_correct), total_loss))
        
        if test_correct >= max_acc:
            max_acc= test_correct
            best_model = epoch
        
        torch.save(model, model_save_folder + '/model_final')

        print("Best performed model (epoch)\t%d" % best_model)


parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-train', help='Training dataset', type=str)
parser.add_argument('-test', help='Validation dataset', type=str)
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=80)
parser.add_argument('-modeldir', help='Folder for trained models', type=str, default='MODEL/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-sample2id', help='Sample to ID mapping file', type=str)
parser.add_argument('-numclass', help='Number of classes', type=str)
parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=3)
parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=3)

parser.add_argument('-exp', help='Expressoin information for samples', type=str)

print("Start....")

# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=3)

# load input data
train_data, sample2id_mapping = prepare_train_data(opt.train, opt.test, opt.sample2id)
gene2id_mapping = load_mapping(opt.gene2id)
print('Total number of genes = %d' % len(gene2id_mapping))

exp_features = np.genfromtxt(opt.exp, delimiter=',')


num_samples = len(sample2id_mapping)
num_genes = len(gene2id_mapping)
num_class = np.array(int(opt.numclass))


# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)


# load the number of hiddens #######
num_hiddens_genotype = np.array(int(opt.genotype_hiddens))


num_hiddens_final = np.array(int(opt.final_hiddens))
#####################################


CUDA_ID = opt.cuda
print("CUDA_ID", CUDA_ID)

train_model(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, opt.modeldir, opt.epoch, opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_final, exp_features, num_class, CUDA_ID)
