# DNASeqClassifier

This miniproject is to utilize two existing DL framework TensorFlow and Nervana Neon to classify the [Molecular Biology (Splice-junction Gene Sequences) Data Set](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29).

Considerations when building up the model and the steps of codes are introduced below. 

## Data preprocessing and dataset split
1. One hot encoder
Since the DNA sequence specifies 8 different input classes, and each category is equally important, no order difference, the input classes are transformed with one-hot encoder:
```
	"A":[0,0,0,0,0,0,0,1],
	"G":[0,0,0,0,0,0,1,0],
	"C":[0,0,0,0,0,1,0,0],
	"T":[0,0,0,0,1,0,0,0],
	"D":[0,0,0,1,0,0,0,0],
	"N":[0,0,1,0,0,0,0,0],
	"S":[0,1,0,0,0,0,0,0],
	"R":[1,0,0,0,0,0,0,0]}
```
It is noticed that the 4 classes are the combination of the other four: 
```
D: A or G or T 
N: A or G or C or T 
S: C or G 
R: A or G
```
So, the result of "OR" is tried out in the experiment, shown below, but does not bring the performance improvement. As a result, the prior one-hot encode (60x8 array) is selected in the experiment.
```
    "A":[0,0,0,1],
    "G":[0,0,1,0],
    "C":[0,1,0,0],
    "T":[1,0,0,0],
    "D":[1,0,1,1],
    "N":[1,1,1,1],
    "S":[0,1,1,0],
    "R":[0,0,1,1]}
```

2. Data filter/clean-up
Exception is checked when converting to one-hot decode. No wrong characters shown from the data set.

3. Training/test split
Data set is shuffled around and split into training and test sets. 20% of total data is used for testing. 
StratifiedShuffleSplit from sklearn package is used to shuffle and split the data.

## Neural network model selection

Usually people use RNN to analyze DNA sequence due to its nature of variant length. However, it is noticed that the DNA sequence in this data set is fixed length: all 60 pairs, so DNN, CNN can be considered to train this fixed length dataset. In this experiment, all DNN, CNN and RNN are exercised. 
Overall CNN provides the best performance. Some more details are explained for CNN and RNN below. 

1. CNN
Even though CNN is widely used on image recognition, this DNA sequence classification can leverage it too. After expanding the data to one-hot encode, each sample is a 60x8 array, similar to a small rectangular image. In this CNN model, the layers are: 
Conv layer (32 feature depth) => pooling => Conv layer (64 feature depth) => pooling => fully connection (64) => dropout => fully connection (16) => fully connection (3: output)

2. RNN





## Hyperparameter selection
4. Regularzation
1. Learning rate
2. Number of hidden layers
3. Node in each layer
5. Activation function 
6. Optimizor 
7. Mini-batch size
8. Loss function