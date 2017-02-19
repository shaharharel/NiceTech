import tensorflow as tf
import numpy as np
import datetime
import model
import params
import random
import utils
import copy

def get_next_batch(train_data,target_data):
    train_batch = []
    target_batch = []
    for t in range(params.batch_size):
       while True:
            idx =[random.choice(train_data.keys()) for i in range(params.batch_size)][0]
            if len(target_data[idx])>=params.number_of_steps:
                continue
            train = copy.copy(train_data[idx])
            target = copy.copy(target_data[idx])
            for i in range(params.number_of_steps - int(float(len(train))/params.num_of_features)):
                train = np.concatenate((train,np.zeros(params.num_of_features)+7))
                target.append(33)
            train = np.asarray(np.split(train,params.number_of_steps))
            target = np.asarray(target)
            #print idx + ' Sequence Length ' + str(len(target_data[idx]))
            train_batch.append(train)
            target_batch.append(target)
            break
    return train_batch,target_batch

def run_training(train_data,target_data,load=False , load_session = None):
    with tf.Graph().as_default():
        print("Starting building graph " + str(datetime.datetime.now()))
        batch_placeholders = tf.placeholder(tf.float32, shape=(None,params.number_of_steps,params.num_of_features))
        target_batch_placeholders = tf.placeholder(tf.int32, shape=(None,params.number_of_steps))
        loss,probabilities = model.inference(batch_placeholders,target_batch_placeholders)
        training = model.training(loss, params.learning_rate)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.trainable_variables())
        if load is True:
           saver.restore(sess, load_session)
           print "Restored!!!"
        for k in range(1, params.num_iters):
            print("Starting iter " + str(k) + " " + str(datetime.datetime.now()))
            data_batch , target_batch = get_next_batch(train_data,target_data)
            print target_batch
            feed_dict = {batch_placeholders: data_batch,target_batch_placeholders:target_batch}
            _, loss_value , prob= sess.run([training, loss,probabilities], feed_dict=feed_dict)
            print [("{0:.3f}".format(np.argmax(prob[i][0])),"{0:.3f}".format(np.max(prob[i][0]))) for i in range(params.number_of_steps)]
            print [("{0:.3f}".format(np.argsort(prob[i])[0][-2]),"{0:.3f}".format(np.sort(prob[i])[0][-2])) for i in range(params.number_of_steps)]
            print [("{0:.3f}".format(np.argsort(prob[i])[0][-3]), "{0:.3f}".format(np.sort(prob[i])[0][-3])) for i in range(params.number_of_steps)]
            print loss_value
            if k % params.save_per_iter == 0 or k==10:
              saver.save(sess, params.output_path + str(k) + '.sess')

if __name__ == '__main__':
    data = utils.load_json_file('sample.json')
    target = utils.load_json_file('sample_target.json')
    train_data = dict(data.items()[:3*len(data) / 4])
    test_data = dict(data.items()[3*len(data) / 4:])
    train_target = dict(target.items()[:3*len(data) / 4])
    test_target = dict(target.items()[3*len(data) / 4:])

    run_training(train_data,train_target)
