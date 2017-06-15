# DNASeqClassifier

This miniproject is to utilize two existing DL framework TensorFlow and Nervana Neon to classify the [Molecular Biology (Splice-junction Gene Sequences) Data Set](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequences%29).

Considerations when building up the model and the steps of codes are introduced below. The rough outline is: 
1. Folder/File introduction
2. Data preprocessing and dataset preparation 
3. Neural network model selection
4. Considerations in building the models
5. Hyperparameter selection
6. Conclusion

## Folder/File introduction
./Data: stores the input data

./Image: stores the captured Tensorboard images from the experiments

./Tensorflow: the folder where implementation is stored. The extensive model configuration is done on TF. 
* DNASeqClf.ipynb: The final selected model
* DNASeqClf_model_tuning.ipynb: the codes to tune the models, do hyper parameters search, with tensorboard log support

./Neon: neon version of implementation
* DNASeqClf_neon.py: get it working on the last minute. :) This is leveraging the IMDB example, with LSTM model, with data input adjusted to fit the DNA data. 

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

Some more details are listed below. 

* **DNN**: 

Hyper parameter search was exercised on DNN, and the result shows one hidden layer is sufficent: 
```
Fully connected layer (200 nodes) => dropout => fully connected layer (3: output)
```

* **CNN**: 

After expanding the data to one-hot vector, each sample is a 60x8 array, similar to a small rectangular image. Thus, this DNA sequence classification can leverage CNN in fact. Hyper parameter search was exercised on the model. The selected structures are: 
```
Conv layer (64 feature depth) => pooling => 
	fully connected layer (64 nodes) => dropout => 
		fully connected layer (16 nodes) => 
			fully connected layer (3: output)
```

* **RNN**: 

Basic RNN cells (vanilla RNN) is used in the experiment. LSTM or GRU are found providing the similar performance while taking more computation. Perhaps it is due to the small sample size. A FC layer is connected after RNN layer to generate the target labels. Time step is set as 60 for 60 DNA slices in each sample. More details of the layers are listed here: 
```
RNN layer (60 time steps, 64 cells) => dropout => fully connected layer (3: output) 
```

## Considerations in building the models

* **Overfitting and Regularzation :** The data set is relatively small: around 3k samples. Overfit was found in the early experiment tests. Several adjustments were done to avoid overfitting including simplify model, adding dropout layer and apply weigth regularization:

    * **Add dropout layer:** Dropout layer is added in both CNN and RNN model after the dense layer (affine layer term used in neon). Dropout rate 0.5. Without Dropout layer, the test accuracy stuck around ~92% while training accuracy keeps improving. With dropout layer, test accuracy continus climbing up with traing accuracy till ~96%.
    
    * **Apply weight regularization:** L2 regularization is applied to the weights in Convolution layer and dense layer. Slight improvement on overfitting on top of dropout layer. 
    
    * Result shows the above regularization successfully mitigated overfitting. 

* **Loss function:** Cross entropy lost function is used here for the classification. Regularization loss is added during training process. 

* **Weight initialization:** Xaiver weight initialization is applied on the conv and dense layer, to adaptively adjust the weight to the appropriate range for the layer. Slight improvement is observed. 

* **Activation function:** Softmax activation function is used in the last layer of the model, since this is a multi-class classification problem, and softmax can nicely represent the class probability distribution. Relu activation function is used in the middle layers for its simiplicity (a max function is sufficient) and help convergence on training. 


## Hyperparameter selection

* **Learning rate**
	Multiple learning rate were examined in hyper parameter search: exponential decayed learning rate, 1E-3, 1E-4, and 1E-5, mapping to lr_0 ~ lr_3 respectively in the below figure. In decayed learning rate in this test, starter learning rate is set as 0.01 and decay rate as 0.5, i.e. reduce into half each time, and reduce to ~1E-5 in 

	The below test shows learning rate 1E-5 (orange color) gives slow converging speed, and not converged within 3000 epochs. Learning rate 1E-3 (yelloe color) supplies some level of variation (the right most diagram). Learning rate 1E-4 does not converage as fast as the decayed learning rate. As a result, the exponential decayed learning rate is used in the experiments.

	![Learning rate selection](https://github.com/QuentinQingLi/DNASeqClassifier/blob/master/Images/Learning_rate_selection.png)

* **Number of hidden layers**
	Layer structure is searched through hyper parameter search for the optimal result in a compromise between classification performance and speed. The below figure is captured from the layer selection from CNN model. Layer structure was traversed the permutation of 1-2 convolution layers and 1-3 fully connected layers. 

	The below figure is "relative" on the horizon axis, representing the relative time consumed in each run. The longer curve means the run taking longer. The result shows the structure of 1 conv layer and 3 FC layer can be an optimal selection, since speed wise, it save \~1/3 execution time comparing to 2 conv and 3 FC layers, also accuracy wise, they are all very close, and this 1conv, 3FC structure slightly better on accuracy.
	![CNN model layer comparison](https://github.com/QuentinQingLi/DNASeqClassifier/blob/master/Images/Accuracy_cnn_model_comparison.png)

	Similarly, a hyper parameter search over learning rate and layers on DNN is also done to find the appropriate hidden layer number. The result is shown in the below figure. It is found that two fully connected layers (one hidden layer) can sufficiently provide the similar performance with more hidden layers and training time is close to half of 4 fully connected layer.  
	![CNN model layer comparison](https://github.com/QuentinQingLi/DNASeqClassifier/blob/master/Images/dnn_hparam_search_full.jpg)

* **Number of hidden RNN cells**
	Multiple cell numbers in RNN layer were tried out in the experiment to determine to an appropriate one for the problem. The test result is shown in the below figure. Smaller cell count, 32 in the test, cannot achieve good a good performance, referring to the orange line. 64 cells and 150 cells can contribute the similar accuracy while 64 cell model takes only 1/3 of execution time of 150 cells, referring to the small time relative figure in the left bottom.
	![CNN model layer comparison](https://github.com/QuentinQingLi/DNASeqClassifier/blob/master/Images/rnn_node_num_selection.jpg)

## Conclusion
DNA data is normally sequential data, but this data set turns out to be fixed length. This facilitated a chance to try out DNN and CNN. The 3 models (DNN, CNN and RNN) provides similar performance after 2K epochs 95%~97%, where CNN is slightly better, but it could be all because of the small data set. If DNA sequence has strong correlation in the adjacent DNA slices in science, RNN could be a better choice in the real-world usage. 

In the expriments and this report, the major design items were discussed. Selection of some of the hyper parameters are introduced also. There must be something I could have missed or wrong in this TF implementation and my description above, but I hope this mini-project can demonstrate my basic understanding on deep learning.  

BTW, neon version of LSTM is much much faster than TF on CPU. Actually my MAC is a Haswell system, two generation older than my windows laptop, a Skylake. But neon runs like flying. :) 
