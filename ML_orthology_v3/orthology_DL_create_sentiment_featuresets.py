#!/usr/bin/env python
# coding: utf-8

# In[7]:


from Bio import SeqIO
from Bio.Data import IUPACData 
import pickle
import tensorflow as tf


# In[8]:


def one_hot_matrix(labels,C):
    
    C = tf.constant(C,name="C")
    one_hot_matrix = tf.one_hot(labels,C,axis=1)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot


# In[14]:


def read_pairs_file(file_in):
    orthoID = {}
    my_index = 0
    my_cluster = {}
    current_index = 0
    
    for line in open(file_in, mode='r'):
        words = line.split()
        
        if words[3].rstrip() == 'n/a': continue
        
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


# In[10]:


def fasta_to_features(input_file,ortho_cluster_IDs,clustertotalnum,out_file_name="features.csv"):
    outfile = open(out_file_name,"w+")
    
    with open(input_file, buffering=100000, mode='r') as handle:
        for fasta in SeqIO.parse(handle,'fasta'):
            name, sequence = fasta.id, str(fasta.seq)
            if name in ortho_cluster_IDs:
                cluster_num = ortho_cluster_IDs[name]
                cluster_num_one_hot = one_hot_matrix([cluster_num],clustertotalnum)
                
                outfile.write('\t'.join([name,sequence,str(cluster_num)]))
                outfile.write("\n")
            
    outfile.close()
    return()


# In[5]:


myOrthoIDs,myclusternum = read_pairs_file('CENH3_DMR6_LUCA-CHLRE00002_orthologues_pairs.txt')  
fasta_to_features('CENH3_DMR6_LUCA-CHLRE00002_orthologues.fa',myOrthoIDs,myclusternum,
                  "features_CENH3_DMR6_LUCA-CHLRE00002_orthologues.csv")


# In[ ]:


myOrthoIDs,myclusternum = read_pairs_file('oma-pairs-viridiplantae.txt')  
fasta_to_features('oma-seqs-viridiplantae.fa',myOrthoIDs,myclusternum,
                  "features_oma-seqs-viridiplantae.csv")


# In[ ]:




