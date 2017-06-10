# DNASeqClassifier

This miniproject is to utilize two existing DL framework TensorFlow and Nervana Neon to classify the [Molecular Biology (Splice-junction Gene Sequences) Data Set](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29).

Considerations when building up the model and the steps of codes are introduced below. 

## Data preprocessing and dataset split
* **One hot encode**: Since the DNA sequence specifies 8 different input classes, and each category is equally important, no order difference, the input classes are transformed with one-hot encoder:
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

* **Data filter/clean-up**: Exception is checked when converting to one-hot decode. No wrong characters shown from the data set.

* **Training/test split**: Data set is shuffled around and split into training and test sets. 20% of total data is used for testing. 
StratifiedShuffleSplit from sklearn package is used to shuffle and split the data.

## Neural network model selection

RNN is usually used to analyze DNA sequence due to the nature of variant length in Gene analysis. However, it is noticed that the DNA sequence input in this data set is fixed length: all 60 pairs, so DNN, CNN can be considered to train this fixed length dataset as well. 

In this experiment, DNN, CNN and RNN are all exercised. Overall CNN provides the best performance from the experiment. However, if the strong correlation exists between the slice junction gene class and the adjacency relationship between Gene pairs, RNN will be still the best choice to use to dig out the sequential features from the gene sequence. 

Some more details are explained for CNN and RNN below. 

* **CNN**: 

After expanding the data to one-hot vector, each sample is a 60x8 array, similar to a small rectangular image. Thus, this DNA sequence classification can leverage CNN too. In this CNN model, the layers are: 
```
Conv layer (32 feature depth) => pooling => 
	Conv layer (64 feature depth) => pooling => 
		fully connection (64 nodes) => dropout => 
			fully connection (16 nodes) => fully connection (3: output)
```

* **RNN**: 

## Considerations in building the models

* **Overfit and Regularzation**
The data set is relatively small: around 3k samples. Overfit was found in the early experiment tests. Regularization was then added to resolve the overfit situation. 
	* Dropout layer: Dropout layer is added in both CNN and RNN model after the dense layer (affine layer term used in neon). Dropout rate 0.5. With Dropout layer added, the test accuracy is improved from ~92% to ~96%
	* Weight regularization: L2 regularization is applied to the weights in Convolution layer and dense layer. Slight improvement on overfitting on top of dropout layer. 

> Implementation notes: So far I haven't found a direct way to apply weight regularization to Tensorflow RNN layer. Looks like weight regularization has to go through direct tensor variables. Due to the time limitation, it is not yet done. 

Result shows the above regularization successfully mitigated overfitting. But RNN result still shows overfitting. Needs to continue explore the solutions.

* Loss function: 

* Weight initialization



## Hyperparameter selection

* Learning rate
* Number of hidden layers
* Node in each layer
* Activation function 
* Optimizor 
* Mini-batch size
