import os
import cv2
import numpy as np
import tqdm
import tensorflow as tf
from layer import *
from network import *
from load import *

def get_points(bounds):
    points = []
    mask = []
    for b in bounds:
        
        print(b)
        
        b = [int(x) for x in b]
        mid_y = (b[0] + b[2])/2
        mid_x = (b[1] + b[3])/2
        
        x1 = int(mid_x - LOCAL_SIZE/2)
        if x1 < 0:
            x1 = 0
        elif x1 > IMAGE_SIZE - LOCAL_SIZE:
            x1 = IMAGE_SIZE - LOCAL_SIZE
        
        y1 = int(mid_y - LOCAL_SIZE/2)
        if y1 < 0:
            y1 = 0
        elif y1 > IMAGE_SIZE - LOCAL_SIZE:
            y1 = IMAGE_SIZE - LOCAL_SIZE
    
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])
        
        p1 = b[0]
        q1 = b[1]
        p2 = b[2]
        q2 = b[3]
        
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)
    
    
    return np.array(points), np.array(mask)

# Hyperparameters
IMAGE_SIZE = 128
LOCAL_SIZE = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
PRETRAIN_EPOCH = 100

x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
is_training = tf.placeholder(tf.bool, [])

model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)

global_step = tf.Variable(0, name='global_step', trainable=False)
epoch = tf.Variable(0, name='epoch', trainable=False)

opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)

d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Load train and test data.
train_data, test_data = load()

step_num = int(len(train_data) / BATCH_SIZE)

# Load model
if tf.train.get_checkpoint_state('./backup'):
    saver = tf.train.Saver()
    saver.restore(sess, './backup/latest')

while True:
    sess.run(tf.assign(epoch, tf.add(epoch, 1)))
    print('epoch: {}'.format(sess.run(epoch)))
    
    np.random.shuffle(train_data)
    
    # Completion
    if sess.run(epoch) <= PRETRAIN_EPOCH:
        g_loss_value = 0
        for i in tqdm.tqdm(range(step_num)):
            train_batch = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            x_batch = np.array([i[0] for i in train_batch])
            x_batch = np.array([a / 127.5 - 1 for a in x_batch])
            points_batch, mask_batch = get_points([i[1] for i in train_batch])
            
            _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
            g_loss_value += g_loss
    
        print('Completion loss: {}'.format(g_loss_value))
        
        np.random.shuffle(test_data)
        test_batch = test_data[:BATCH_SIZE]
        x_batch = np.array([i[0] for i in test_batch])
        x_batch = np.array([a / 127.5 - 1 for a in x_batch])
        points_batch, mask_batch = get_points([i[1] for i in test_batch])
        
        completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
        sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
        cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
        
        saver = tf.train.Saver()
        saver.save(sess, './backup/latest', write_meta_graph=False)
        if sess.run(epoch) == PRETRAIN_EPOCH:
            saver.save(sess, './backup/pretrained', write_meta_graph=False)

    # Discrimitation
    else:
        g_loss_value = 0
        d_loss_value = 0
        for i in tqdm.tqdm(range(step_num)):
            train_batch = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            x_batch = np.array([i[0] for i in train_batch])
            x_batch = np.array([a / 127.5 - 1 for a in x_batch])
            points_batch, mask_batch = get_points([i[1] for i in train_batch])
            
            _, g_loss, completion = sess.run([g_train_op, model.g_loss, model.completion], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
            g_loss_value += g_loss
            
            local_x_batch = []
            local_completion_batch = []
            for i in range(BATCH_SIZE):
                x1, y1, x2, y2 = points_batch[i]
                local_x_batch.append(x_batch[i][x1:x2, y1:y2, :])
                local_completion_batch.append(completion[i][x1:x2, y1:y2, :])
            local_x_batch = np.array(local_x_batch)
            local_completion_batch = np.array(local_completion_batch)
            
            _, d_loss = sess.run(
                                 [d_train_op, model.d_loss],
                                 feed_dict={x: x_batch, mask: mask_batch, local_x: local_x_batch, global_completion: completion, local_completion: local_completion_batch, is_training: True})
            d_loss_value += d_loss
        
        print('Completion loss: {}'.format(g_loss_value))
        print('Discriminator loss: {}'.format(d_loss_value))
        
        np.random.shuffle(test_data)
        test_batch = test_data[:BATCH_SIZE]
        x_batch = np.array([i[0] for i in test_batch])
        x_batch = np.array([a / 127.5 - 1 for a in x_batch])
        points_batch, mask_batch = get_points([i[1] for i in test_batch])
        
        completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
        sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
        cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
        
        saver = tf.train.Saver()
        saver.save(sess, './backup/latest', write_meta_graph=False)

#if __name__ == '__main__':
#    x_train, x_test = load()
#    print(x_train.shape)
#    print(x_test.shape)

