# ## Install libraries 
#     sudo -H pip install biopython
#     sudo -H pip install numpy
#     sudo -H pip install tensorflow
#     
# You also need to save <span style="color:brown">orthology_DL_create_sentiment_featuresets.ipynb</span> program in <span style="color:brown">orthology_DL_create_sentiment_featuresets.py</span> format in the same directory of this file. We load it like a library to use the functions.


from orthology_DL_create_sentiment_featuresets import create_feature_sets_and_labels,protein_seq_to_binary
import tensorflow as tf
import pickle
import numpy as np
from Bio import SeqIO
from Bio.Data import IUPACData 


# ## Creating TRAIN set and TEST set
# 
# <span style="color:blue">**creat_feature_sets_and_labels**</span> function converts the features into proper formatted Train set and Test set.
# 
# **inputs:**
# 
# 1.Paires file is provided in this format:
# 
#    <span style="color:orange">geneID [tab] geneID [tab] type_of_correlation [tab] OMA-orthology-group-number</span>
# 
# Sample:
# 
#     ORYNI25992	ARATH00066	1:1	488382
#     ORYNI25992	THECC03451	1:1	488382
#     ORYPU22376	ORYRU26544	1:1	488382
# 
# 2.Proteins fasta file , which includes multiple protein sequences.
# 
# Sample:
# 
#     >MARPO07937	
#     MAAMTMTRPNDPFTSDIARFHEVKGAKDEPFAELTRYHEVRGVREMVESFRVREVPSYYV
#     KPVNERRFTPPSAAVLSMEQQIPCIDLEALSGQELLSAIANACRDWGFFQVLNHGLPSQL
#     VQNMAKQSSEFFAQPLEEKMKCSTPARVSGPVHFGGGGNRDWRDVLKLNCAPASIVAKEY
#     WPQRPAGFRDTMEEYSSQQQALAIRLLKLISESLGLESNYLVAACGEPKVVMAINHYPPC
#     PDPSLTMGIKAHSDPNTITMLLQDDVGGLQVFKEDRWIDVRPLPNALVINVGDQLQILSN
#     GKYSSCLHRVVNNNRQARTSIATFFSPAHACIIGPAPGLVDEVNPAIYPNIVYADYIKAF
#     YTQALGPNNKNGGYLAGIELHRRYNCYTSSSSISS
# 
# 3.The size of the Test set is by default 10% (0.1) of the data, you can change it if you wish.
# 
# 


#train_x,train_y,test_x,test_y,l_seq_size = create_feature_sets_and_labels('CENH3_DMR6_orthologues_pairs.txt','CENH3_DMR6_orthologues.fa',0.1)
#train_x,train_y,test_x,test_y = create_feature_sets_and_labels('CENH3_DMR6_LUCA-CHLRE00002_orthologues_pairs.txt','CENH3_DMR6_LUCA-CHLRE00002_orthologues.fa',0.1)
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('oma-pairs-viridiplantae.txt','oma-seqs-viridiplantae.fa',0.1)

# ## Setting variables


# Number of nodes in each NN hidden layer
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

# Defining the fix size of protein sequence (used in Test set and Train set) to use for input protein queries
l_seq_size = int(len(np.array(test_x)[0])/len(IUPACData.extended_protein_letters))

# Number of orthology clusters
n_classes = len(np.array(test_y)[0])     #2 or 3 or ...

# Batch size and Epoch size for training the NN
batch_size = 32   #100
hm_epochs = 10    #1000

# Initializing X and Y
x = tf.placeholder('float')
y = tf.placeholder('float')

# Initializing NN layers
hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

# The Saver class adds ops to save and restore variables to and from checkpoints. 
saver = tf.train.Saver()
tf_log = 'tf.log'


# ## Neural Network Model
# 
# 
# ![alt text](./NN-3hl.jpg "NN model")


def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output


# ## Training Neural Network
# 
# The trained model is saved to <span style="color:brown">./model.ckpt*</span>.


def train_neural_network(x):
    my_model_save_path = "./model.ckpt"
    prediction = neural_network_model(x)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,"./model.ckpt")
            epoch_loss = 1
            
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                epoch_loss += c
                i+=batch_size
                
            
            my_model_save_path = saver.save(sess, "./model.ckpt")
            
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch +=1
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("\nModel saved in path: %s " % my_model_save_path)
        print('\nAccuracy:',accuracy.eval({x:test_x, y:test_y}))

    
    return()



train_neural_network(x)


# ## Converting query protein fasta file to a binary table
# 
# <span style="color:blue">**fasta_input_to_features**</span> function converts a protein fasta file to a binary table.
# 
# How it works?
# 
# Each protein sequence is converted to an arrays of binary arrays. For that we use "protein_seq_to_binary" function.
# but then we flatten this 2 dimentional array. so what is left, is a long binary sequence - (0,1)s . All protein sequences in the fasta file will have the same binary length = l_seq_size x 26.
# 
# So for each fasta file, fasta_input_to_features returns a 2 dimentional array (all_seq_features): 
# 
#     Number of rows  = number of sequences in the fasta file
#     Number of columns = 1 (a binary list representing the protein sequence)
# 
# and an array of gene IDs (all_names).These IDs will be used for proper result report.


def fasta_input_to_features(input_fasta_file,lseqsize=23776):
    fasta_sequences = SeqIO.parse(open(input_fasta_file),'fasta')
    
    all_seq_features = []
    all_names = []
    
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequence_features = protein_seq_to_binary(sequence,lseqsize)
        sequence_features_flat = sequence_features.flatten()
        all_seq_features.append(list(sequence_features_flat))
        all_names.append(name)
        
        
    return(all_seq_features,all_names)


# ## Using the trained-tested model
# 
# Now it is time to use our model. <span style="color:blue">**use_neural_network**</span> function gets a query protein fasta file and returns predicted OMA group for each proteins sequence based on our model.
# 
# The program reads the model. Predicts the orthology cluster for each protein sequence. converts the cluster IDs to OMA groups and returns results in a dictionary format.
# 
# See the example below.


def use_neural_network(in_fasta_file,lseqsize=l_seq_size): # there can be multiple sequences in one fasta file
    prediction = neural_network_model(x)
    
    input_features, input_names = fasta_input_to_features(in_fasta_file,lseqsize)
    my_cluster_inv = pickle.load( open( "OMAgroupID-newClusterNumbers.pickle", "rb" ) )    
    #print(my_cluster_inv)    
    
    output={}
    i=0
    
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"model.ckpt")
        for item in input_features:
            features = item
            features = np.array(list(features))
            result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
            omacluster = my_cluster_inv[result[0]]
            #print("\n* ",input_names[i],"is in cluster",result,"= oma group",omacluster) 
            output[input_names[i]]=omacluster
            i += 1
    
    return(output)
    

##use_neural_network("./CHLRE00002.fa",l_seq_size)
#print(use_neural_network("./CHLRE00002.fa"))


## import the inspect_checkpoint library
#from tensorflow.python.tools import inspect_checkpoint as chkp

## print all tensors in checkpoint file
#chkp.print_tensors_in_checkpoint_file("./model.ckpt", tensor_name='', all_tensors=True)

