import tensorflow as tf
import params
from tensorflow.python.ops import rnn, rnn_cell

def inference(batch_placeholder,target_placeholder):
    W = tf.Variable(tf.random_uniform([params.lstm_dim,params.num_of_labels],name='W'))
    b = tf.Variable(tf.zeros([1,params.num_of_labels]),name='b')
    initial_state = tf.zeros([params.batch_size, params.lstm_dim])
    #numpy_state = initial_state.eval()
    loss, predictions = time_series_LSTM_loss(W,b,batch_placeholder,target_placeholder)
    return loss, predictions

def createLSTM(hidden_dim):
    #return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_dim),0.5,0.5)
    return tf.nn.rnn_cell.BasicLSTMCell(hidden_dim,state_is_tuple=False)

def time_series_LSTM_loss(W,b, example,target):
    count = 0
    lstm_cell = createLSTM(params.lstm_dim)
    state = tf.zeros([params.batch_size, params.lstm_dim*2])
    #example=tf.squeeze(example)
    probabilities = []
    predictions=[]
    loss = 0.0
    # Permuting batch_size and n_steps
    x = tf.transpose(example, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, params.num_of_features])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, params.number_of_steps, x)
    with tf.variable_scope("lstm") as scope:
        for session in range(10):
            if count >0:
                scope.reuse_variables()
            count+=1
            output, state = lstm_cell(x[session], state)
            logits = tf.matmul(output, W) + b
            probabilities.append(logits)
            predictions.append(tf.nn.softmax(logits))
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target[:,session], name='sparseSoftmaxLoss')
            #(probabilities, target[i])
    return loss, predictions

def training(loss, learningRate):
    print("Begin training")
    return tf.train.AdagradOptimizer(learningRate).minimize(loss)
