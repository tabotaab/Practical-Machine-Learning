from orthology_DL_create_sentiment_featuresets import create_feature_sets_and_labels,protein_seq_to_binary
from orthology_DL_train_neural_network import use_neural_network
import sys

if len(sys.argv)<2:
    print("Usage:\tpython3 ./orthology_DL_use_neural_network.py CHLRE00002.fa")
else:
    result = use_neural_network(sys.argv[1])
    for k in result:
        print("\n* ",k,"is in oma group",result[k]) 
            
        

