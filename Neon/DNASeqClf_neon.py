from future import standard_library
standard_library.install_aliases()  # triggers E402, hence noqa below
from neon import logger as neon_logger  # noqa
from neon.backends import gen_backend  # noqa
from neon.data import ArrayIterator  # noqa
from neon.initializers import Uniform, GlorotUniform, Array  # noqa
from neon.layers import GeneralizedCost, Affine, Dropout, LSTM, RecurrentSum, Recurrent  # noqa
from neon.models import Model  # noqa
from neon.optimizers import Adagrad  # noqa
from neon.transforms import Logistic, Tanh, Softmax, CrossEntropyMulti, Accuracy  # noqa
from neon.util.argparser import NeonArgparser, extract_valid_args  # noqa
from neon.callbacks.callbacks import Callbacks  # noqa
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


'''
Load the data
'''
def load_data(file_path) :
    df = pd.read_csv(file_path, header=None)
    df.columns = ['classlabel', 'name', 'sequence']
    df.tail()
    
    return df


'''
Pre-process the dataset
  - Apply one-hot-encoding to input data
  - Take 20% as test data
'''
def preprocess_data(df) :
    
    # Encoding class labels
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    #print("y:",y)
    
    # Encoding sequence
    # Here we use one hot encoding to encode the character in DNA sequence. 
    # So each dna sequence is converted to a 60x8 2D array 
    def Seq2Vec(seq):
        s = str(seq).strip()
        CharDict = { "A":[0,0,0,0,0,0,0,1],
                     "G":[0,0,0,0,0,0,1,0],
                     "C":[0,0,0,0,0,1,0,0],
                     "T":[0,0,0,0,1,0,0,0],
                     "D":[0,0,0,1,0,0,0,0],
                     "N":[0,0,1,0,0,0,0,0],
                     "S":[0,1,0,0,0,0,0,0],
                     "R":[1,0,0,0,0,0,0,0]}
        return np.asarray([CharDict[c] for c in s], dtype=np.float32).flatten()

    df['seqvec'] = df['sequence'].apply(Seq2Vec)
    X = np.vstack(df['seqvec'].values)
    print("Total samples:", X.shape[0])  #X: [sample,60,8]
    
    # Split the data set into training/test set
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    print("Training samples: ", X_train.shape[0], 
          "Test samples: ", X_test.shape[0])
    
    return X_train, y_train, X_test, y_test



def model_rnn(X_train, y_train, X_test, y_test, num_epochs):
    # hyperparameters
    hidden_size = 64
    nclass = 3
    

    # make train dataset
    train_set = ArrayIterator(X_train, y_train, nclass=nclass)
    # make valid dataset
    valid_set = ArrayIterator(X_test, y_test, nclass=nclass)
    
    # initialization
    init_glorot = GlorotUniform()
    
    
    # define layers
    layers = [
        LSTM(hidden_size, init_glorot, activation=Tanh(), gate_activation=Logistic(),
             reset_cells=True),
        RecurrentSum(),
        Dropout(keep=0.5),
        Affine(nclass, init_glorot, bias=init_glorot, activation=Softmax())
    ]
    
    # set the cost, metrics, optimizer
    cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
    metric = Accuracy()
    model = Model(layers=layers)
    optimizer = Adagrad(learning_rate=0.01)
    
    # configure callbacks
    callbacks = Callbacks(model, eval_set=valid_set, **args.callback_args)
    
    # train model
    model.fit(train_set,
              optimizer=optimizer,
              num_epochs=num_epochs,
              cost=cost,
              callbacks=callbacks)
    
    # eval model
    neon_logger.display("Train Accuracy - {}".format(100 * model.eval(train_set, metric=metric)))
    neon_logger.display("Test Accuracy - {}".format(100 * model.eval(valid_set, metric=metric)))




# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--rlayer_type', default='lstm',
                    choices=['bilstm', 'lstm', 'birnn', 'bibnrnn', 'rnn'],
                    help='type of recurrent layer to use (lstm, bilstm, rnn, birnn, bibnrnn)')

args = parser.parse_args(gen_be=False)
args.batch_size = 100

print(args)

# setup backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# Load data
df = load_data("../data/splice.data")

# Preprocess data and split training and validation data
X_train, y_train, X_test, y_test = preprocess_data(df)
    
# RNN model
model_rnn(X_train, y_train, X_test, y_test, num_epochs=50)
    



