# DNASeqClassifier

This miniproject is to utilize two existing DL framework TensorFlow and Nervana Neon to classify the [Molecular Biology (Splice-junction Gene Sequences) Data Set](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29).

Considerations when building up the model and the steps of codes are introduced below. The rough outline is: 
1. Data preprocessing and dataset preparation 
2. Neural network model selection
3. Considerations in building the models
4. Hyperparameter selection
5. Folder/File introduction

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

After expanding the data to one-hot vector, each sample is a 60x8 array, similar to a small rectangular image. Thus, this DNA sequence classification can leverage CNN in fact. Hyper parameter search was exercised on the model. The selected structures are: 
```
Conv layer (64 feature depth) => pooling => 
	fully connected layer (64 nodes) => dropout => 
		fully connected layer (16 nodes) => 
			fully connected layer (3: output)
```

* **RNN**: 

Basic RNN architecture is used here, no LSTM or GRU utilized. It is connected to a fully connected layer to generate the target labels. More details of the layers are listed here: 
```
RNN layer (64 RNN cells) => dropout => fully connected layer (3: output) 
```

## Considerations in building the models

* **Overfitting and Regularzation :** The data set is relatively small: around 3k samples. Overfit was found in the early experiment tests. Several adjustments were done to avoid overfitting including simplify model, adding dropout layer and apply weigth regularization:

    * **Simplify the model:** 150 RNN cells were initially used in RNN layer in RNN model. The goal was trying to store long enough (>x2 60 pairs) int the RNN cells. It turns out overfitting quickly, test accuracy stucks at 89% while training accuracy goes >99%. Reducing the RNN cells to 64 dramatically helped the overfitting. 
    
    * **Add dropout layer:** Dropout layer is added in both CNN and RNN model after the dense layer (affine layer term used in neon). Dropout rate 0.5. Without Dropout layer, the test accuracy stuck around ~92% while training accuracy keeps improving. With dropout layer, test accuracy continus climbing up with traing accuracy till ~96%.
    
    * **Apply weight regularization:** L2 regularization is applied to the weights in Convolution layer and dense layer. Slight improvement on overfitting on top of dropout layer. 
    
    * Result shows the above regularization successfully mitigated overfitting. 

	> Implementation notes: So far I haven't found a direct way to apply weight regularization to Tensorflow RNN layer. Looks like weight regularization has to go through direct tensor variables. Due to the time limitation, it is not yet done. 

* **Loss function:** Cross entropy lost function is used here for the classification. Regularization loss is added during training process. 

* **Weight initialization:** Xaiver weight initialization is applied on the conv and dense layer, to adaptively adjust the weight to the appropriate range for the layer. Slight improvement is observed. 


## Hyperparameter selection

* **Learning rate**
	Multiple learning rate were examined in hyper parameter search: exponential decayed learning rate, 1E-3, 1E-4, and 1E-5, mapping to lr_0 ~ lr_3 respectively in the below figure. In decayed learning rate in this test, starter learning rate is set as 0.01 and decay rate as 0.5, i.e. reduce into half each time, and reduce to ~1E-5 in 

	The below test shows learning rate 1E-5 (orange color) gives slow converging speed, and not converged within 3000 epochs. Learning rate 1E-3 (yelloe color) supplies some level of variation (the right most diagram). Learning rate 1E-4 does not converage as fast as the decayed learning rate. As a result, exponentially decayed learning rate is selected.

	![Learning rate selection](https://github.com/QuentinQingLi/DNASeqClassifier/blob/master/Images/Learning_rate_selection.png)

* **Number of hidden layers**
	Layer structure is searched through hyper parameter search for the optimal result in a compromise between classification performance and speed. The below figure is captured from the layer selection from CNN model. Layer structure was traversed the permutation of 1-2 convolution layers and 1-3 fully connected layers. 

	The below figure is "relative" on the horizon axis, representing the relative time consumed in each run. The longer curve means the run taking longer. The result shows the structure of 1 conv layer and 3 FC layer can be an optimal selection, since speed wise, it save \~1/3 execution time comparing to 2 conv and 3 FC layers, also accuracy wise, they are all very close, and this 1conv, 3FC structure slightly better on accuracy.
	![CNN model layer comparison](https://github.com/QuentinQingLi/DNASeqClassifier/blob/master/Images/Accuracy_cnn_model_comparison.png)


* Node in each layer
* Activation function 
* Optimizor 
* Mini-batch size

## Folder/File introduction
./Data: stores the input data

./Image: stores the captured Tensorboard images from the experiments

./Tensorflow: the folder where implementation is stored
* DNASeqClf.ipynb: The final selected model
* DNASeqClf_model_tuning.ipynb: the codes to tune the models, do hyper parameters search, with tensorboard log support

./Neon: empty. The original plan was to implement the solution on neon also, but find out not enough time. :(

