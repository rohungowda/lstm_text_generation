import pandas as pd
import string
import torch
import torchtext
import random
import tracemalloc
from torch.utils.data import Dataset, DataLoader, random_split 
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Possible Ideas
# idea of maybe adding context words for what the song should be about??
# use print(number_to_words_dictionary[y_i[y].item()])to get the word guess

# utility functions and variables



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')





# if lstm is not working lets try with a bidireciton lstm and then a  Bert embedding

# Grid Search
batch_sizes = [32,64] # try with 64 and 128
hidden_lstm_sizes = [256] # if it does better wiht the 512 lets use 1024 and a higher value as well, should give us 4 concrete valuesto work with.
hidden_linear_sizes = [1024] # 512 
epochs = 15 # max number of epochs before overfitting
drop_percentages = [0.25] # better values with increased dropout percentage, 0.25 working well so lets keep it
learning_rates = [.01]
weight_decays = [0] # 0, 0.10, 0.25


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, hidden1dim, output_dim, dropout_dim,sequence_len):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.batch = nn.BatchNorm1d(sequence_len * hidden_dim * 2) # withbidirectinal lstm
        self.hidden = nn.Linear(sequence_len * hidden_dim * 2, hidden1dim)
        self.output = nn.Linear(hidden1dim, output_dim)
        self.drop = nn.Dropout(dropout_dim)

    def forward(self, data):
        lstm_out, _ = self.lstm(data)
        flatten_out = torch.flatten(lstm_out, start_dim=1)
        batch_out = self.batch(flatten_out)
        hidden_out = self.hidden(batch_out)
        drop_out = self.drop(hidden_out)
        logits = self.output(drop_out)
        return logits

class data_set(Dataset): 
    def __init__(self,X_data,Y_data): 
        self.X_data = X_data
        self.Y_data = Y_data

  
    def __len__(self): 
        return len(self.X_data) 
  
    def __getitem__(self, index):
        return (self.X_data[index],self.Y_data[index])

early_stopper = EarlyStopping(patience=3, min_delta=0.0025)
loss_fn = torch.nn.CrossEntropyLoss()

def loop(training_data_loader, validation_data_loader, testing_data_loader,optimizer,epochs,model):
    epoch_losses = {}
    early_stopper.reset()
    earlyStopping = False
    for epoch in range(epochs):
        if earlyStopping:
            break
        epoch_losses[epoch] = []
        print("Epoch: " + str(epoch))
        #print()
        #print("--------------------------------------------------------------------------")
        #print("TRAINING")
        #print("--------------------------------------------------------------------------")
        check_size = len(training_data_loader) // 5
        total_loss = 0
        running_loss = 0

        model.train()
        for i,batch in enumerate(training_data_loader):
            x_i, y_i = batch[0],batch[1]

            # forward pass
            logits = model(x_i)
            loss = loss_fn(logits, y_i)
            total_loss += loss
            running_loss += loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Total Training Average Loss: " + str((total_loss/len(training_data_loader)).item()))
        epoch_losses[epoch].append(total_loss/len(training_data_loader))
        # store training data + Save Model
        #print()


        model.eval()
        #print("--------------------------------------------------------------------------")
        #print("VALIDATION")
        #print("--------------------------------------------------------------------------")
        check_size = len(validation_data_loader) // 5
        total_loss = 0
        running_loss = 0

        for i,batch in enumerate(validation_data_loader):
            x_i, y_i = batch[0],batch[1]

            # forward pass
            with torch.no_grad():
                logits = model(x_i)
            loss = loss_fn(logits, y_i)
            total_loss += loss
            running_loss += loss

            # check early stopping here?

        print("Total Validation Average Loss: " + str((total_loss/len(validation_data_loader)).item()))
        epoch_losses[epoch].append(total_loss/len(validation_data_loader))
        scheduler.step(total_loss/len(validation_data_loader))
        if early_stopper(total_loss/len(validation_data_loader)):
            print(f"Early stopping at epoch {epoch}")
            earlyStopping = True
        
        # store validation data
        #print()


        #print("--------------------------------------------------------------------------")
        #print("TESTING")
        #print("--------------------------------------------------------------------------")
        check_size = len(testing_data_loader) // 5
        total_loss = 0
        running_loss = 0

        for i,batch in enumerate(testing_data_loader):
            x_i, y_i = batch[0],batch[1]

            # forward pass
            with torch.no_grad():
                logits = model(x_i)
            loss = loss_fn(logits, y_i)
            total_loss += loss
            running_loss += loss

            # check early stopping here?


        print("Total Testing Average Loss: " + str((total_loss/len(testing_data_loader)).item()))
        epoch_losses[epoch].append(total_loss/len(testing_data_loader))
        # store Testing data
        #print()
    return epoch_losses

# utility function and variables

# read data, pre process data
df = pd.read_csv('spotify_millsongdata.csv')
df = df.loc[df['artist'] == 'ABBA']
df = df['text']

table = str.maketrans('', '', string.punctuation)

words_set = set()
number_to_words_dictionary = {}
words_to_numbers_dictionary = {}

def preProcess(x):
    x = x.strip()
    x = x.lower()
    x = x.splitlines()

    for i in range(len(x)):
        x[i] = x[i].lower()
        x[i] = x[i].translate(table)
        x[i] = x[i].split()
        x[i].append('<EOS>')
        words_set.update(x[i])
    
    return x

print("Preprocessing staring...")
data = []
for i in range(len(df)):
    data.extend(preProcess(df[i]))

# creates words dictionary, this will be the output nodes for the softmax function
words_set = list(sorted(words_set))
for i,word in enumerate(words_set):
    number_to_words_dictionary[i] = word
    words_to_numbers_dictionary[word] = i



max_len = max(len(x) for x in data)
# Load GloVe embeddings
glove = torchtext.vocab.GloVe(name="6B", dim=50)

#convert data into X and y data, with X being embeddings and y as numbers
X = []
Y = []

# convert into word embeddings

# torch size to be (max_size, 50)

for i in range(len(data)):
    update = torch.tensor([])
    for j in range(len(data[i]) - 1):
        x_embedding = glove[data[i][j]]
        y_embedding = words_to_numbers_dictionary[data[i][j+1]]
        temp,update = torch.cat((update,x_embedding)),torch.cat((update,x_embedding))
        temp = torch.reshape(temp,(j+1,50))
        X.append(temp.clone())
        Y.append(y_embedding)
# right now padding sequence to be a tensor combined each time, we kind of want more of a list of tensors?

# Pad the sequences
padded_X = pad_sequence(X, batch_first=True)



# split training data into training, validation and testing


# creates dataset
dataset = data_set(padded_X,Y)

train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size = len(dataset) - train_size - test_size

# Split the dataset
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

print("Preprocessing Completed ... ")


print("GRID SEARCH STARTING ...")

classes_size = len(number_to_words_dictionary)

#accuracy = Accuracy(task="multiclass", num_classes=classes_size)

# STARTS THE WHOLE LOOP
sequence_length = padded_X.size()[1]
input_dim = padded_X.size()[2]

tracemalloc.start()
with open("readme.txt", "w") as f:
    for batch in batch_sizes:

    # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)

        for hidden_lstm in hidden_lstm_sizes:
            for hidden_linear in hidden_linear_sizes:
                for drop_p in drop_percentages:
                    
                    # creates model
                    model = LSTM(input_dim, hidden_lstm, hidden_linear, classes_size, drop_p,20)

                    for lr in learning_rates:
                        for wd in weight_decays:
                    # runs training loop
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
                            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
                            name = f"batch: {batch}, hidden_lstm: {hidden_lstm}, hidden_linear: {hidden_linear}, drop_percentage: {drop_p}, learning_rate: {lr}, weight_decay: {wd}"
                            print(name)
                            print("----------------------------------------------------------------------------------------------------------------------------------------------")
                            val= loop(train_loader,val_loader,test_loader,optimizer,epochs=epochs,model=model)
                            f.write(name + "\n")
                            for key,value in val.items():
                                f.write(f"| At epoch: {key} training_loss = {value[0]} validation_loss = {value[1]} testing_loss = {value[2]}" + "\n")
                                print(f"|   At epoch: {key}     |   training_loss = {value[0]}        |   validation_loss = {value[1]}        |   testing_loss = {value[2]}        |")
                            f.write("\n")
                            val = None
                            del val
                            print(tracemalloc.get_traced_memory())
                            del optimizer
                            del scheduler
                    del model
        train_loader = None
        val_loader= None
        test_loader = None
        del train_loader
        del val_loader
        del test_loader
                
tracemalloc.stop()
print("GRID SEARCH COMPLETED")

print()

'''
with open("readme.txt", "w") as f:
    for key,epoch_losses in parameters.items():
        f.write(key + "\n")
        for key,value in epoch_losses.items():
            print(f"|   At epoch: {key}   |   training_loss = {value[0]}   |   validation_loss = {value[1]}   |   testing_loss = {value[2]}   |")
    for line in lines:

'''

        



'''
scale data to proabably one artist's songs for now
create each lyric as follows
['look', 'at', 'her', 'face', 'its', 'a', 'wonderful', 'face','EOS']

[look,pad,pad,pad,pad,pad,pad,pad,pad] [at]
[look at, pad,pad,pad,pad,pad,pad,pad] [her]
[look at her,pad,pad,pad,pad,pad,pad] [face]
[look at her face its,pad,pad,pad,pad,pad] [a]
[look at her face its a,pad,pad,pad,pad] [wonderfull]
[look at her face its a wonderfull,pad,pad,pad] [face]
[look at her face its a wonderfull face,pad,pad] [EOS] -> this ends the sentance generation.

pad each of the X data after EOS to max length for verses
then feed each word throughs the embeddings.
Input and hidden size should be (batch size, max_length,300)
feed through fully connected network with max_length * 300
before feeding through output layer with nodes for probabilities for each word in dictionary

for dealing with predicting we stop the model when it reaches an end of sentance token.


                        x_test,y_test = dataset[5]
                        x_test = x_test.reshape(1,20,50)

                        result = model(x_test)
                        ind = torch.argmax(result)
                        print("----")
                        print(ind.item())
                        print(y_test)

'''


# first get a rough idea of the data, visualize the average sentance length.