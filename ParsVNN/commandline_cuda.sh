#!/bin/bash
codedir="/Users/yijwang-admin/Documents/Research/GCN_Structure/VNN_Learning/AD_Application/ParsVNN"
inputdir="/Users/yijwang-admin/Documents/Research/GCN_Structure/VNN_Learning/AD_Application/BulkRNAseq"
ontfile="/Users/yijwang-admin/Documents/Research/GCN_Structure/VNN_Learning/AD_Application/BulkRNAseq/ImmuneSystem_Pathway_Reactome.txt"
gene2idfile="/Users/yijwang-admin/Documents/Research/GCN_Structure/VNN_Learning/AD_Application/BulkRNAseq/rosmap_gene2id.txt"
sample2idfile="/Users/yijwang-admin/Documents/Research/GCN_Structure/VNN_Learning/AD_Application/BulkRNAseq/rosmap_sample2id.txt"
expfile="/Users/yijwang-admin/Documents/Research/GCN_Structure/VNN_Learning/AD_Application/BulkRNAseq/rosmpa_exp.txt"
modeldir="/Users/yijwang-admin/Documents/Research/GCN_Structure/VNN_Learning/AD_Application/BulkRNAseq/MODEL"

#foldid=$1
cudaid=$1

#modeldir=MODEL_$foldid
#mkdir $modeldir

python -u $codedir/train_parsvnn.py  -onto $ontfile -gene2id $gene2idfile -sample2id $sample2idfile -train $inputdir/rosmap_train.txt -test $inputdir/rosmap_test.txt -model $modeldir -exp $expfile -genotype_hiddens 6 -final_hiddens 6 -numclass 4 

