import tensorflow as tf
import numpy as np
import datetime
import model
import params
import random
#import  DataPreProcess
import evaluation
import pickle
import utils
import copy

def get_next_batch(train_data,target_data):
   while True:
    idx =[random.choice(train_data.keys()) for i in range(params.batch_size)][0]
    #idx='d13c4ad5-1fc6-41fb-8f7c-1e915a2f753a'
    if len(target_data[idx])>=10:
        continue
    train = copy.copy(train_data[idx])
    target = copy.copy(target_data[idx])
    for i in range(params.number_of_steps - int(float(len(train))/params.num_of_features)):
        train = np.concatenate((train,np.zeros(params.num_of_features)))
        target.append(32)
    train = np.expand_dims(np.asarray(np.split(train,params.number_of_steps)),0)
    target = np.expand_dims(np.asarray(target),0)
    print idx
    return train,target

def run_training(path_data,target_path,iterLoss = 100,load=False , load_session = None):
    print("Loading training data...")
    train_data = utils.load_json_file(path_data)
    target_data = utils.load_json_file(target_path)
    with tf.Graph().as_default():
        print("Starting to build graph " + str(datetime.datetime.now()))
        batch_placeholders = tf.placeholder(tf.float32, shape=(None,params.number_of_steps,params.num_of_features))
        target_batch_placeholders = tf.placeholder(tf.int32, shape=(None,params.number_of_steps))
        loss,probabilities = model.inference(batch_placeholders,target_batch_placeholders)
        training = model.training(loss, params.learning_rate)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.trainable_variables())
        if load is True:
           saver.restore(sess, 'sessions/sess1/40000.sess')
           print "Restored!!!"
        for i in range(1, params.num_iters):
            print("Starting iter " + str(i) + " " + str(datetime.datetime.now()))
            data_batch , target_batch = get_next_batch(train_data,target_data)
            print target_batch
            feed_dict = {batch_placeholders: data_batch,target_batch_placeholders:target_batch}
            _, loss_value , prob= sess.run([training, loss,probabilities], feed_dict=feed_dict)
            print loss_value
            print [(np.argmax(prob[i][0]),np.max(prob[i][0])) for i in range(10)]
            if i % params.save_per_iter == 0:
              saver.save(sess, params.output_path + str(i) + '.sess')

if __name__ == '__main__':
    #sequences = utils.load_json_file('sample.json')
    #target = utils.load_json_file('sample_target.json')
    run_training('sample.json','sample_target.json')