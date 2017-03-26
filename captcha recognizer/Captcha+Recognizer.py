
# coding: utf-8

# In[40]:

import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


# In[41]:

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


# In[42]:

# Generate captcha picture
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image():
    image = ImageCaptcha()
    
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


# In[43]:

# Convert colorful image to gray image
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # above is a quick way to transform, but is an official way
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('Captcha has maximum length as 4')

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map') 
        return k
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector
    

def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)



# In[44]:

def get_next_batch(batch_size = 128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    
    # in case sometimes the image size vary with (60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image
            
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
        
    return batch_x, batch_y

# define captcha nn skelton
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    # 3 conv layer
#     w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
#     b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
#     conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides= [1, 1, 1, 1], padding = 'SAME'), b_c1))
#     conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv1 = tf.nn.dropout(conv1, keep_prob)

#     w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
#     b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
#     conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c2, strides= [1, 1, 1, 1], padding = 'SAME'), b_c2))
#     conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv2 = tf.nn.dropout(conv2, keep_prob)
    
#     w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
#     b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
#     conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c3, strides= [1, 1, 1, 1], padding = 'SAME'), b_c3))
#     conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#     conv3 = tf.nn.dropout(conv3, keep_prob)
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    
    # Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    
    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    
    return out

# captcha recognizer nn structure
def train_crack_captcha_cnn():
    output    = crack_captcha_cnn()
    loss    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = Y))
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict   = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)
            
            # calculate accuracy every 100 step
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 0.75})
                print(step, acc)
            
                # if accuray is greater than 50, store result and finish training
                if acc > 0.85:
                    saver.save(sess, "./model/crack_captcha.model", global_step=step)
                    break
            step += 1
    

def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "crack_capcha.model-4300")

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1.})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)
#     output = crack_captcha_cnn()
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         # saver.restore(sess, "/Users/zhengyangqiao/Desktop/Neuron Network/captcha recognizer/crack_capcha.model-4300")
# #         saver.restore(sess, 'crack_capcha.model-4300')
# #         new_saver = tf.train.import_meta_graph('crack_capcha.model-4300.meta')
# #         new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#         predict = tf.argmax(tf.reshape(output,[-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
#         text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
#         text = text_list[0].tolist()
#
#         return text


# In[45]:

if __name__ == '__main__':
    train = 0
    if train == 0:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        text, image = gen_captcha_text_and_image()
        print("Captcha size : ", image.shape)
        
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        print("maximum captcha string: ", MAX_CAPTCHA)
        
        char_set = number
        CHAR_SET_LEN = len(char_set)
        
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
        # dropout
        keep_prob = tf.placeholder(tf.float32)
        
        train_crack_captcha_cnn()
    if train == 1:
        # number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # IMAGE_HEIGHT = 60
        # IMAGE_WIDTH =  160
        # char_set = number
        # CHAR_SET_LEN = len(char_set)
        # text, image = gen_captcha_text_and_image()
        #
        # f = plt.figure()
        # ax = f.add_subplot(111)
        # ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
        #
        # MAX_CAPTCHA = len(text)
        # image = convert2gray(image) / 255
        #
        # X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        # Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
        # keep_prob = tf.placeholder(tf.float32)
        #
        # predict_text = crack_captcha(image)
        # print("correct: () forcast: ()".format(text, predict_text))

        text, image = gen_captcha_text_and_image()
        image = convert2gray(image)
        image = image.flatten() / 255

        IMAGE_HEIGHT = 60
        IMAGE_WIDTH =  160
        char_set = number
        CHAR_SET_LEN = len(char_set)
        MAX_CAPTCHA = len(text)

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32)

        predict_text = crack_captcha(image)
        print("正确: {}  预测: {}".format(text, predict_text))



