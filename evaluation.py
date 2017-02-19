import tensorflow as tf
import model
import random
import params
import copy
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report
import utils
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def adjust_test_batch(point,target):
        #print ' Sequence Length ' + str(len(target))
        test = copy.copy(point)
        target = copy.copy(target)
        for i in range(params.number_of_steps - int(float(len(test)) / params.num_of_features)):
            test = np.concatenate((test, np.zeros(params.num_of_features) + 7))
            target.append(33)
        test = np.asarray(np.split(test, params.number_of_steps))
        target = np.asarray(target)
        return test, target

def evaluation_metrics(actual,predicted,iter):
    cm = confusion_matrix(actual,predicted)
    print cm
    target_names = ["channel "+str(i) for i in sorted(set(actual).union(set(predicted)))]
    #heat_map(cm,target_names,iter)
    #with open('visual/report.txt','a') as file:
    #    file.write(iter+'\n'+'\n')
    #    file.write(classification_report(actual,predicted,target_names=target_names)+'\n'+'\n'+'\n')

def heat_map(conf_arr,channels,iter):
    norm_conf=[]
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')
    width, height = conf_arr.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), channels,rotation=90)
    plt.yticks(range(height), channels)
    plt.savefig('visual/confusion_matrix_'+iter+'.png', format='png')

def run_evaluationMul(test_data,test_target_data, session_path):  # 'sessions/sess1/40000.sess'
    print("Loading test data...")
    total_actual_per_seq = {}
    total_pred_per_seq = {}
    for i in range(params.number_of_steps):
        total_actual_per_seq[i]=[]
        total_pred_per_seq[i]=[]
    total_actual = []
    total_pred = []
    with tf.Graph().as_default():
        batch_placeholders = tf.placeholder(tf.float32, shape=(None, params.number_of_steps, params.num_of_features))
        target_batch_placeholders = tf.placeholder(tf.int32, shape=(None, params.number_of_steps))
        loss, probabilities = model.inference(batch_placeholders, target_batch_placeholders)
        sess = tf.Session()
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, session_path)
        current = 0
        test_keys = test_data.keys()
        while current<len(test_keys):
            test_batch = []
            test_target = []
            for i in range(params.batch_size):
                if current == len(test_keys):
                    leftOvers = params.batch_size - len(test_target)
                    for left in range(leftOvers):
                        test_batch.append(np.zeros([params.number_of_steps,params.num_of_features]))
                        test_target.append(np.zeros([params.number_of_steps])+33)
                    break
                while len(test_target_data[test_keys[current]])>= params.number_of_steps:
                    current+=1
                    if current == len(test_keys):
                        break
                    continue
                if current == len(test_keys):
                    continue
                test_b, test_t = adjust_test_batch(test_data[test_keys[current]],test_target_data[test_keys[current]])
                test_batch.append(test_b)
                test_target.append(test_t)
                current+=1
            feed_dict = {batch_placeholders: test_batch, target_batch_placeholders: test_target}
            loss_value, prob = sess.run([loss, probabilities], feed_dict=feed_dict)
            for k in range(params.batch_size):
                eliminated = []
                for i,j in enumerate(test_target[k]):
                    if j==33:
                        continue
                    if j==0:
                        eliminated.append(i)
                    else:
                        total_actual.append(j)
                        total_actual_per_seq[i].append(j)
                for j in range(params.number_of_steps):
                    if len(total_pred_per_seq[j])<len(total_actual_per_seq[j]) and j not in eliminated:
                        prediction = np.argmax(prob[j][k])
                        total_pred.append(prediction)
                        total_pred_per_seq[j].append(prediction)
        print "Total Evaluation"
        evaluation_metrics(total_actual,total_pred,session_path.split('/')[-1])

        # print "Per position evaluation"
        # for i in range(params.number_of_steps-1):
        #     print "position " +str(i)
        #     print evaluation_metrics(total_actual_per_seq[i],total_pred_per_seq[i],session_path.split('/')[-1])

        return

if __name__ == '__main__':
    data = utils.load_json_file('sample.json')
    target = utils.load_json_file('sample_target.json')
    train_data = dict(data.items()[:3*len(data) / 4])
    test_data = dict(data.items()[3*len(data) / 4:])
    train_target = dict(target.items()[:3*len(data) / 4])
    test_target_data = dict(target.items()[3*len(data) / 4:])

    for session in os.listdir('output'):
        if not session.startswith('session') or not session.endswith('.sess'):
            continue
        print "test evaluation " + session
        run_evaluationMul(train_data,train_target,'output/'+session)
