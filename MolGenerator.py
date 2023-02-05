import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Concatenate
from keras import regularizers

def Pair_Molecules(sdf_path, similarity_threshold=0.9):
    list_of_mol = []
    for file in os.listdir(sdf_path):
        if file.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(os.path.join(sdf_path, file))
            for molecules in suppl:
                list_of_mol.append(molecules)
    substructure = FindMCS(list_of_mol).smartsString
    matches = [molecules.GetSubstructMatches(Chem.MolFromSmarts(substructure)) for mol in list_of_mol]
    return matches

#http://gdb.unibe.ch/downloads/
smfile = "/home/ABTLUS/mario.neto/Desktop/MoleculeGenerator/gdb11_size11.smi"
#size11
data = pd.read_csv(smfile, delimiter="\t", names= ["smiles","No", "Int"])
smiles_train, smiles_test = train_test_split(data["smiles"], random_state=42)

#The SMILES must be vectorized to one-hot encoded arrays
#RNN's will be trained with batch mode 
charset = set("".join(list(data.smiles))+"!E") #O método join() junta todos os elementos de um array
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in data.smiles]) + 5

def vectorize(smiles):
        one_hot =  np.zeros((smiles.shape[0], embed , len(charset)),dtype=np.int8)
        for i,smile in enumerate(smiles):
            one_hot[i,0,char_to_int["!"]] = 1
            for j,c in enumerate(smile):
                one_hot[i,j+1,char_to_int[c]] = 1
            one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
        return one_hot[:,0:-1,:], one_hot[:,1:,:]

X_train, Y_train = vectorize(smiles_train.values)
X_test, Y_test = vectorize(smiles_test.values)

print(smiles_train.iloc[0])
plt.matshow(X_train[0].T)

#set the Keras objects
#import the dimensions for input and output
#Additionally the number of LSTM 
#cell to use for the decoder and encoder is specified 
#and the latent dimension is specified.
input_shape = X_train.shape[1:]
output_dim = Y_train.shape[-1]
#64 LSTM cells is used to read the input SMILES string
latent_dim = 64
lstm_dim = 64

unroll = False
encoder_inputs = Input(shape=input_shape)
encoder = LSTM(lstm_dim, return_state=True,
                unroll=unroll)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
states = Concatenate(axis=-1)([state_h, state_c])
neck = Dense(latent_dim, activation="relu")
neck_outputs = neck(states)
