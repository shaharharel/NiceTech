import tensorflow as tf
import params
from tensorflow.python.ops import rnn, rnn_cell

def inference(batch_placeholder,target_placeholder):
    W = tf.Variable(tf.random_uniform([params.lstm_dim,params.num_of_labels],name='W'))
    b = tf.Variable(tf.zeros([1,params.num_of_labels]),name='b')
    loss, predictions = time_series_LSTM_loss(W,b,batch_placeholder,target_placeholder)
    return loss, predictions

def createLSTM(hidden_dim):
    #return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_dim),0.5,0.5)
    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim,state_is_tuple=False)
    return rnn_cell.MultiRNNCell([lstm] * 2,state_is_tuple=False)

def time_series_LSTM_loss(W,b, example,target):
    count = 0
    stacked_lstm = createLSTM(params.lstm_dim)
    state =stacked_lstm.zero_state(params.batch_size, tf.float32)
    probabilities = []
    predictions=[]
    loss = 0.0
    x = tf.transpose(example, [1, 0, 2])
    x = tf.reshape(x, [-1, params.num_of_features])
    x = tf.split(0, params.number_of_steps, x)
    with tf.variable_scope("lstm") as scope:
        for session in range(params.number_of_steps):
            if count >0:
                scope.reuse_variables()
            count+=1
            output, state = stacked_lstm(x[session], state)
            logits = tf.matmul(output, W) + b
            probabilities.append(logits)
            predictions.append(tf.nn.softmax(logits))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target[:,session], name='sparseSoftmaxLoss'))
    return loss, predictions

def training(loss, learningRate):
    print("Begin training")
    return tf.train.AdagradOptimizer(learningRate).minimize(loss)
