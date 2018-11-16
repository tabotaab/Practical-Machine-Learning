# ## Install libraries 
#     sudo -H pip install biopython
#     sudo -H pip install numpy
#     sudo -H pip install tensorflow
#     
# Optional (it's nice for you to have these in general! , haha) :
# 
#     sudo -H pip install scipy
#     sudo -H pip install scikit-learn
#     sudo -H pip install matplotlib
#     sudo -H pip install pandas
#     




from Bio import SeqIO
from Bio.Data import IUPACData 
import numpy as np
import random
import pickle
import tensorflow as tf


# ## Parsing orthology pairs file 
# Paires file is provided in this format:
# 
#    <span style="color:orange">geneID [tab] geneID [tab] type_of_correlation [tab] OMA-orthology-group-number</span>
# 
# Sample:
# 
#     ORYNI25992	ARATH00066	1:1	488382
#     ORYNI25992	THECC03451	1:1	488382
#     ORYPU22376	ORYRU26544	1:1	488382
# 
# <span style="color:blue">**read_pairs_file**</span> function parses this file, and returns:
# 1. Orthology dictionary made of gene IDs (keys) and their new orthology cluster number (values)
# 2. Number of existing orthology clusters
# 
# It also saves pairs of the <span style="color:orange">OMA cluster IDs ~ new ID numbers</span> in <span style="color:brown">OMAgroupID-newClusterNumbers.pickle</span>




def read_pairs_file(file_in):
    orthoID = {}
    my_index = 0
    my_cluster = {}
    current_index = 0
    
    for line in open(file_in, mode='r'):
        words = line.split()
        
        # we first convert OMA cluster IDs (4th column) to our own ID numbers (my_index starting from 0)
        if words[3].rstrip() in my_cluster:
            current_index = my_cluster[words[3].rstrip()]
        else:
            my_cluster[words[3].rstrip()] = my_index
            current_index = my_index
            my_index += 1
            
        # then we relate each gene ID to this new cluster number
        orthoID[words[0]] = current_index
        orthoID[words[1]] = current_index        
    
    # we save a pickle file storing : OMA cluster IDs ~ new ID numbers (we need it later)
    with open('OMAgroupID-newClusterNumbers.pickle','wb+') as f:
        pickle.dump({v: k for k, v in my_cluster.items()}, f, protocol=pickle.HIGHEST_PROTOCOL)        
    
    return(orthoID,len(my_cluster.keys()))


# ## Longest protein sequence
# I checked the longest plant protein sequence size in 2 ways:
# 1. [link to NCBI query](https://www.ncbi.nlm.nih.gov/protein/?term=20000%3A30000%5BSequence+Length%5D)
# 
#    Plants -> Accession: PNW77743.1 has 23859 aa protein
#    
#    
# 2. python3 fasta_seq_size.py oma-seqs-viridiplantae.fa > tmp
# 
#    CHLRE14650 has the longest sequence with size 23776
# 
# so I decided to give the default longest size "23776". 
# But to prevent unnecessary calculations I also added a function which calculates only the longest sequence size in the input fasta file "longest_seq_size()". And I try to use this number instead of the longest possible sequence size in general.
# 
# <span style="color:blue">**longest_seq_size**</span> function parses fasta input file, and returns:
# 1. length of the longest protein sequence in the file
# 2. name of the longest protein sequence in the file



def longest_seq_size(file_in):
    max_size = 0
    max_name = ""
    for seq_record in SeqIO.parse(open(file_in, mode='r'), 'fasta'):
        name, sequence = seq_record.id, str(seq_record.seq)
        if (max_size < len(sequence)) :
            max_size = len(sequence)
            max_name = name
    return(max_size,max_name)


# ## Converting a protein sequence to binary
# 
# <span style="color:blue">**protein_seq_to_binary**</span> function converts any protein sequence to arrays of binary values. 
# 
# How it works?
# 
# We have 26 IUPAC protein letters. For each letter in the protein sequence we consider an array of 26 values, which the i-th value is 1 and the rest are 0. For instance, whenever we see a 'K', the 9th value in the array is set to 1 and the rest are set to 0. 
# 
# Sequences can have different length. But for this algorithm we decided to work with fixed size. Therefore, the program gets a fixed value as input (l_seq_size) which is always >= sequence length. 
# 
# Like that for each sequence we get an array (l_seq_size) of arrays (26 values) - 2 dimentional array. If the fixed value of l_seq_size is larger than sequence length, the extras will have only 0 in their array. 
# 
# See the example below.



def protein_seq_to_binary(in_seq,l_seq_size=23776):
    all_protein_letters = list(IUPACData.extended_protein_letters) # 26 letters
        
    features = np.zeros((l_seq_size,len(all_protein_letters)))
    #features = np.zeros((len(in_seq),len(all_protein_letters)))

    for i in range(0,len(in_seq)):
        aminoacid = list(in_seq)[i]
        if aminoacid.upper() in all_protein_letters:
            index_value = all_protein_letters.index(aminoacid.upper())
            features[i][index_value] = 1
            
    return(features)




#protein_seq_to_binary("K",2) 


# ## Converting a protein fasta to a binary table
# 
# <span style="color:blue">**fasta_to_features**</span> function converts a protein fasta file to a binary table.
# 
# How it works?
# 
# Each protein sequence is converted to an arrays of binary arrays. For that we use "protein_seq_to_binary" function.
# but then we flatten this 2 dimentional array. so what is left, is a long binary sequence - (0,1)s . All protein sequences in the fasta file will have the same binary length = l_seq_size x 26.
# 
# The function also keeps the gene ID and the cluster number of that gene ID.
# 
# So for each fasta file, fasta_to_features returns a 2 dimentional array (all_seq_features): 
# 
#     Number of rows  = number of sequences in the fasta file
#     Number of columns = 3 (first column is gene ID, second column is a binary list representing the protein sequence, third item is the cluster number)
# 
# See the example below.



def fasta_to_features(input_file,ortho_cluster_IDs,l_seq_size=23776):
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
        
    all_seq_features = []
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        if name in ortho_cluster_IDs:
            mylist = []
            mylist.append([name])
            sequence_features = protein_seq_to_binary(sequence,l_seq_size)
            sequence_features_flat = sequence_features.flatten()
            mylist.append(list(sequence_features_flat))
            mylist.append(ortho_cluster_IDs[name])
            all_seq_features.append(mylist)
            
    return(all_seq_features)



#seq_f = fasta_to_features("CHLRE00002.fa",{"CHLRE00002":5},390) # e.g. cluster 5 = 737531 OMA group
#print(np.array(seq_f).shape)
#print(seq_f)


# ## Convering cluster number to one_hot matrix
# 
# The locations represented by indices in indices take value *on_value* (default = 1), while all other locations take value *off_value* (default = 0).
# 
# for instance: 
# 
#     indices = [0, 1, 2]
#     depth = 3
#     tf.one_hot(indices, depth)  # output: [3 x 3]
#     [[1., 0., 0.],
#      [0., 1., 0.],
#      [0., 0., 1.]]
#     
# <span style="color:blue">**one_hot_matrix**</span> function converts any number (label, here we input cluster number) to a one_hot tensor of given depth (C, which is total number of clusters).
# 
# See the example below.



def one_hot_matrix(labels,C):
    
    C = tf.constant(C,name="C")
    one_hot_matrix = tf.one_hot(labels,C,axis=1)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot



#features_seq_f = np.array(seq_f)
#print(list(features_seq_f[:,2][:]))
#print(one_hot_matrix(list(features_seq_f[:,2][:]),6))


# ## Creating TRAIN set and TEST set
# 
# <span style="color:blue">**creat_feature_sets_and_labels**</span> function converts the features into proper formatted Train set and Test set.
# 
# The size of the Test set is by default 10% of the data, but you can change it if you wish.
# 
# I made a function called <span style="color:blue">**test_me**</span> to demonstrate the results.




def create_feature_sets_and_labels(pairs_file,fasta_file,test_size = 0.1):
       
    myOrthoIDs,myclusternum = read_pairs_file(pairs_file)  
    
    l_seq_size,l_seq_name = longest_seq_size(fasta_file)
    
    features = []
    features = fasta_to_features(fasta_file,myOrthoIDs,l_seq_size)
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features)) # Calculating the test size based on input 

    train_x = list(features[:,1][:-testing_size])  # Protein sequences are already converted to one_hot like
    train_y = list(one_hot_matrix(list(features[:,2][:-testing_size]),myclusternum)) # convert cluster number to one_hot format
    test_x = list(features[:,1][-testing_size:])
    test_y = list(one_hot_matrix(list(features[:,2][-testing_size:]),myclusternum))

    return train_x,train_y,test_x,test_y




if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('CENH3_DMR6_orthologues_pairs.txt','CENH3_DMR6_orthologues.fa',0.1)
    # if you want to pickle this data:
    with open('./sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)




def test_me():
    #train_x,train_y,test_x,test_y = create_feature_sets_and_labels('CENH3_DMR6_orthologues_pairs.txt','CENH3_DMR6_orthologues.fa',0.1)
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('CENH3_DMR6_LUCA-CHLRE00002_orthologues_pairs.txt','CENH3_DMR6_LUCA-CHLRE00002_orthologues.fa',0.1)
    print("Size of Train X matrix:",np.array(train_x).shape)
    print("Size of Train Y matrix:",np.array(train_y).shape)
    print("Size of Test X matrix:",np.array(test_x).shape)
    print("Size of Test Y matrix:",np.array(test_y).shape)
    print("==============================================")
    print("First element of Train X matrix:",train_x[0])
    print("First element of Train Y matrix:",train_y[0])
    print("==============================================")
    print("Last (69th) element of Train X matrix:",train_x[68])
    print("Last (69th) element of Train Y matrix:",train_y[68])
    return()




#test_me()

