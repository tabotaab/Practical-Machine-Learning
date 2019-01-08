#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical

from Bio import SeqIO
from Bio.Data import IUPACData 
import csv
import numpy as np
import tensorflow as tf
#import dask.dataframe as dd
#import dask.array as da

#data_path = 'features_CENH3_DMR6_LUCA-CHLRE00002_orthologues.csv'
#data_path = 'features_oma-seqs-viridiplantae_test-7-8.csv'
#data_path = 'features_oma-seqs-viridiplantae_test-4-5-6-7-8.csv'
#data_path = 'features_oma-seqs-viridiplantae_test-4-5-6-7-8-9.csv'
data_path = 'features_oma-seqs-viridiplantae_test-4-5-6-7-8-9-10.csv'



def protein2integer(in_seq):
    
    ## define universe of possible input values
    all_protein_letters = list(IUPACData.extended_protein_letters)
    ## define a mapping of chars to integers 
    ## i+1 beacuse we want to start from integer 1 instead of 0. 0 will be used for padding
    char_to_int = dict((c, i+1) for i, c in enumerate(all_protein_letters))
    int_to_char = dict((i+1, c) for i, c in enumerate(all_protein_letters))
    ## integer encode input data
    integer_encoded = [char_to_int[char] for char in in_seq.upper()]
    
    return(integer_encoded)
    

def make_dataset(in_file):
    with open(in_file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        # get all the rows as a list
        d_set = list(reader)
        # transform data into numpy array
        d_set = np.array(d_set).astype(str)
        
    integer_encoded_proteins = np.array([protein2integer(seq) for seq in d_set[:,1]])
    
    G = d_set[:, 0]
    X = integer_encoded_proteins
    Y = d_set[:, 2].astype(int)
                         
    return(d_set,G,X,Y)

def make_dataset_dask(in_file):
    data = dd.read_csv(in_file,sep='\t', header=None)
    df = data.compute().reset_index(drop=True)
    integer_encoded_proteins = da.from_array([protein2integer(seq) for seq in df.values[:,1]],chunks=1000)
    G = df.values[:,0]
    X = integer_encoded_proteins.compute()
    Y = df.values[:,2].astype(int)
                     
    return(df,G,X,Y)

def make_train_test_set_idea1(G,X,Y):
	
	# here we keep 80% of random indexes in train set and the rest in test set
	
    indices = np.random.permutation(X.shape[0])
    train_size = int(indices.size*0.80)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train, X_test = X[train_idx,], X[test_idx,]  
    y_train, y_test = Y[train_idx,], Y[test_idx,]
    
    return(X_train,y_train,X_test,y_test)

def make_train_test_set_idea2(G,X,Y):
    
    # here we try to keep one item from each cluster in test set
    # the rest goes to train set
    
    test_idx = []
    train_idx = []
    seen_cluster_id = []
    
    for i in range(0,Y.shape[0]):
            if Y[i] in seen_cluster_id :
                train_idx.append(i)
            else:
                test_idx.append(i)
                seen_cluster_id.append(Y[i])
    
    X_train, X_test = X[train_idx,], X[test_idx,]    
    y_train, y_test = Y[train_idx,], Y[test_idx,]
       
    return(X_train,y_train,X_test,y_test)
    
    
def model1(X_train_new, y_train,X_test_new, y_test,in_batch_size=100,in_epochs=10): 
	# RNN: Recurrent Neural Networks + LSTM
    # create the model
    embedding_vecor_length = 4
    model = Sequential()
    model.add(Embedding(num_letters, embedding_vecor_length, input_length=fixed_seq_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(75))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    
    # Convert labels to categorical one-hot encoding & fit the model
    y_train_one_hot_labels = to_categorical(y_train, num_classes=n_classes)
    model.fit(X_train_new, y_train_one_hot_labels, epochs=in_epochs, batch_size=in_batch_size)

    # evaluate the model
    y_test_one_hot_labels = to_categorical(y_test, num_classes=n_classes)
    loss, accuracy = model.evaluate(X_test_new, y_test_one_hot_labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    
    return()


def model2(X_train_new, y_train,X_test_new, y_test,in_batch_size=100,in_epochs=10): 
	# RNN: Recurrent Neural Networks
    # Initializing the Sequential model from KERAS.
    model = Sequential()

    # Creating a 16 neuron hidden layer with Linear Rectified activation function.
    model.add(Dense(16, input_dim=fixed_seq_length, kernel_initializer='uniform', activation='relu'))

    # Creating a 8 neuron hidden layer.
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

    # Adding a output layer.
    model.add(Dense(n_classes, kernel_initializer='uniform', activation='softmax'))
    
    # Compiling the model
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

    print(model.summary())
   
    # Convert labels to categorical one-hot encoding & fit the model
    y_train_one_hot_labels = to_categorical(y_train, num_classes=n_classes)
    model.fit(X_train_new, y_train_one_hot_labels, epochs=in_epochs, batch_size=in_batch_size, verbose=1)

    # evaluate the model
    y_test_one_hot_labels = to_categorical(y_test, num_classes=n_classes)
    loss, accuracy = model.evaluate(X_test_new, y_test_one_hot_labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    
    return()



def one_hot_matrix(labels,C):
    
    C = tf.constant(C,name="C")
    one_hot_matrix = tf.one_hot(labels,C,axis=1)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot


def model3(X_train_new, y_train,X_test_new, y_test, batch_size =100, hm_epochs =100): # CNN: Convolutional Neural Networks
    # Number of nodes in each NN hidden layer
    n_nodes_hl1 = 1500
    n_nodes_hl2 = 1500
    n_nodes_hl3 = 1500
   
    
    train_y = one_hot_matrix(y_train,n_classes)
    test_y = one_hot_matrix(y_test,n_classes)

    # Initializing X and Y
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    # Initializing NN layers
    hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(X_train_new[0]), n_nodes_hl1])),
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


    
    
    l1 = tf.add(tf.matmul(x,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    prediction = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

        
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 1

        while epoch <= hm_epochs:
            epoch_loss = 1
            
            i=0
            while i < len(X_train_new):
                start = i
                end = i+batch_size
                batch_x = np.array(X_train_new[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                epoch_loss += c
                i+=batch_size
                
            
            print('Epoch ',epoch,' out of ',hm_epochs,'- loss:',epoch_loss)

            epoch +=1
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('\nAccuracy:',accuracy.eval({x:X_test_new, y:test_y}) * 100)
    return()


def model4(X_train_new, y_train,X_test_new, y_test,in_batch_size=100,in_epochs=10): # RNN: Recurrent Neural Networks
    
    # Convert labels to categorical one-hot encoding  
    y_train_one_hot_labels = to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot_labels = to_categorical(y_test, num_classes=n_classes)
    X_train_new_one_hot_labels = np.array([to_categorical(x, num_classes=num_letters) for x in X_train_new])
    X_test_new_one_hot_labels = np.array([to_categorical(x, num_classes=num_letters) for x in X_test_new])

    # create the model    
    model = Sequential()
    model.add(LSTM(75,input_shape=X_train_new_one_hot_labels[0].shape,return_sequences=True))
    model.add(LSTM(75))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())  

    # fit the model
    model.fit(X_train_new_one_hot_labels, y_train_one_hot_labels, epochs=in_epochs, batch_size=in_batch_size)

    # evaluate the model
    loss, accuracy = model.evaluate(X_test_new_one_hot_labels, y_test_one_hot_labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    
    return()


def model5(X_train_new, y_train,X_test_new, y_test,in_batch_size=100,in_epochs=10): # RNN: Recurrent Neural Networks
    # LSTM only
    
    # Convert labels to categorical one-hot encoding  
    y_train_one_hot_labels = to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot_labels = to_categorical(y_test, num_classes=n_classes)

    # create the model    
    model = Sequential()
    model.add(Embedding(num_letters, output_dim=fixed_seq_length))
    model.add(LSTM(75,return_sequences=True))
    model.add(LSTM(75))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    # fit the model
    model.fit(X_train_new, y_train_one_hot_labels, epochs=in_epochs, batch_size=in_batch_size)

    # evaluate the model
    loss, accuracy = model.evaluate(X_test_new, y_test_one_hot_labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    
    return()


def model6(X_train_new, y_train,X_test_new, y_test,in_batch_size=100,in_epochs=10): # RNN: Recurrent Neural Networks
    # GRU
    
    # Convert labels to categorical one-hot encoding  
    y_train_one_hot_labels = to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot_labels = to_categorical(y_test, num_classes=n_classes)
    
    # create the model    
    model = Sequential()
    model.add(Embedding(num_letters, output_dim=fixed_seq_length))
    model.add(GRU(128, return_sequences=False))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    # fit the model
    model.fit(X_train_new, y_train_one_hot_labels, epochs=in_epochs, batch_size=in_batch_size)

    # evaluate the model
    loss, accuracy = model.evaluate(X_test_new, y_test_one_hot_labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    
    return()

def use_model(model_json_file="model.json",model_h5_file="model.h5"):
    # load json and create model
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_h5_file)
    print("Loaded model from disk")
 
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    return()
  
def model7(X_train_new, y_train,X_test_new, y_test,in_batch_size=100,in_epochs=10,model_json_file="model.json",model_h5_file="model.h5"): # RNN: Recurrent Neural Networks
    # GRU
    
    # Convert labels to categorical one-hot encoding  
    y_train_one_hot_labels = to_categorical(y_train, num_classes=n_classes)
    y_test_one_hot_labels = to_categorical(y_test, num_classes=n_classes)
    
    # create the model    
    model = Sequential()
    model.add(Embedding(num_letters, output_dim=fixed_seq_length))
    model.add(GRU(128, return_sequences=False))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    # fit the model
    model.fit(X_train_new, y_train_one_hot_labels, epochs=in_epochs, batch_size=in_batch_size)

    # evaluate the model
    #loss, accuracy = model.evaluate(X_test_new, y_test_one_hot_labels, verbose=0)
    #print('Accuracy: %f' % (accuracy*100))
    scores = model.evaluate(X_test_new, y_test_one_hot_labels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0])) # Loss
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) # Accuracy
 
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_h5_file)
    print("Saved model to disk")
    
    return()

dataset, G, X, Y = make_dataset(data_path)
X_train,y_train,X_test,y_test = make_train_test_set_idea2(G,X,Y)

num_letters = len(list(IUPACData.extended_protein_letters)) # = 26 amino acids

#fixed_seq_length = len(max(X, key=len)) # maximum
#fixed_seq_length = (sum(len(X[i,]) for i in range(X.shape[0]))/X.shape[0])  # average
fixed_seq_length = 1000

n_classes = int(np.amax(np.concatenate((y_train,y_test),axis=0))+1)

# truncate and pad input sequences
X_train_new = sequence.pad_sequences(X_train, maxlen=fixed_seq_length, padding='post', truncating='post')
X_test_new = sequence.pad_sequences(X_test, maxlen=fixed_seq_length, padding='post', truncating='post')
  
#model1(X_train_new, y_train, X_test_new, y_test,256,10000)
#model2(X_train_new, y_train, X_test_new, y_test,256,10000)
#model3(X_train_new, y_train, X_test_new, y_test,256,100)
#model4(X_train_new, y_train, X_test_new, y_test,256,50)
#model5(X_train_new, y_train, X_test_new, y_test,256,50)
#model6(X_train_new, y_train, X_test_new, y_test,256,1000)
model7(X_train_new, y_train, X_test_new, y_test,256,800,"GRU_model.json",model_h5_file="GRU_model.h5")
