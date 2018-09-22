import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from network import Network
from load import load_test, load


# Hyperparameters
IMAGE_SIZE = 128
LOCAL_SIZE = 64
BATCH_SIZE = 16

test_data, _= load()
print(len(test_data))
test_data = [x for x in test_data if len(x[1]) == 4]
print(len(test_data))

def test():
    
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, '../backup/latest')

    np.random.shuffle(test_data)
    

    step_num = int(len(test_data) / BATCH_SIZE)

    cnt = 0
    for i in tqdm.tqdm(range(step_num)):
        test_batch = test_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        x_batch = np.array([i[0] for i in test_batch])
        x_batch = np.array([a / 127.5 - 1 for a in x_batch])
        _, mask_batch = get_points([i[1] for i in test_batch])
        completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch[i]
            raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
            masked = raw * (1 - mask_batch[i]) + np.ones_like(raw) * mask_batch[i] * 255
            img = completion[i]
            img = np.array((img + 1) * 127.5, dtype=np.uint8)
            dst = './output/{}.jpg'.format("{0:06d}".format(cnt))
            output_image([['Input', masked], ['Output', img], ['Ground Truth', raw]], dst)


def get_points(bounds):
    points = []
    mask = []
    for b in bounds:
        
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
    

def output_image(images, dst):
    fig = plt.figure()
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.xlabel(text)
    plt.savefig(dst)
    plt.close()


if __name__ == '__main__':
    test()
    
