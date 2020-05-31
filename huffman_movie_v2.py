# coding:utf-8
import tensorflow as tf
import sys
import numpy as np
import os
import random
import collections

import Config_movie as Config
import Model
import Huffman_Encoding


bit_num = sys.argv[1]
bit_num = np.int32(bit_num)
index = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()
file = 'data/movie.txt'
word_all = []
data = open(file, 'r').readlines()
for line in range(len(data)):
    data_line = data[line]
    line_words = data_line.split(' ')
    for word in line_words:
        word_all.append(word)
words = list(set(word_all))     # char vocabulary

data_size, _vocab_size = len(word_all), len(words)
print 'data has %d words, %d unique.' % (data_size, _vocab_size)
word_to_idx = {wo: i for i, wo in enumerate(words)}
idx_to_word = {i: wo for i, wo in enumerate(words)}

idx_unknown = word_to_idx['unknown']

config.vocab_size = len(word_to_idx)
len_of_generation = config.len_of_generation


def pro_start_word(statistics1):
    sel_word_sta = []
    sel_value_sta = []
    for i in range(100):
        k = statistics1[i]
        key = k[0]
        value = k[1]
        sel_word_sta.append(key)
        sel_value_sta.append(value)
    sel_value_sta = np.array(sel_value_sta)
    sel_value_sta = sel_value_sta/float(sum(sel_value_sta))
    start = np.random.choice(sel_word_sta, 1, p=sel_value_sta)
    start_word = start[0]
    while not start_word.islower():
        start = np.random.choice(sel_word_sta, 1, p=sel_value_sta)
        start_word = start[0]
    return start_word


def run_epoch(session, m, data, eval_op, state=None):
    """Runs the model on the given data."""
    x = data.reshape((1, 1))
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op], {m.input_data: x, m.initial_state: state})
    return prob, _state


def main(_):
    os.makedirs('generate/movie', exist_ok=True)
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        config.batch_size = 1
        config.num_steps = 1

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = Model.Model(is_training=False, config=config)

        model_saver = tf.train.Saver()
        print 'model loading ...'
        model_saver.restore(session, config.model_path+'-70')
        print 'Done!'

        start_words = []
        data = open('data/movie.txt', 'r').readlines()
        for i in range(len(data)):
            line_ste = data[i].strip()
            line_ste = line_ste.split(' ')
            start_word = line_ste[0]
            start_words.append(start_word)
        statistics = collections.Counter(start_words)
        statistics1 = sorted(statistics.items(), key=lambda item: item[1], reverse=True)

        bit_stream = open('./bit_stream/bit_stream.txt', 'r').readline()
        outfile = open('./generate/movie/movie_' + str(bit_num) + 'bit' + '_' + index + '.txt', 'w')
        bitfile = open('./generate/movie/movie_' + str(bit_num) + 'bit' + '_' + index + '.bit', 'w')
        bit_index = random.randint(0, 30000)
        count = 0
        while count < 100:
            start_word = pro_start_word(statistics1)
            if start_word == 'unknown':
                continue

            start_idx = word_to_idx[start_word]
            _state = mtest.initial_state.eval()
            test_data = np.int32([start_idx])
            prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
            gen_res = [start_word]
            gen = word_to_idx['unknown']
            while gen == word_to_idx['unknown']:
                gen = np.random.choice(config.vocab_size, 1, p=prob.reshape(-1))
                gen = gen[0]
            # the second word is chose randomly
            test_data = np.int32(gen)
            gen_res.append(idx_to_word[gen])
            bit = ""

            for i in range(len_of_generation - 2):
                if idx_to_word[gen] in ['\n', '']:
                    break
                prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
                p = prob.reshape(-1)
                p[idx_unknown] = 0
                prob_sort = sorted(p)
                prob_sort.reverse()
                word_prob = [prob_sort[i] for i in range(2**int(bit_num))]

                # TODO review2
                # words_prob = []
                # while len(words_prob) < 2 ** int(bit_num):
                #     ind = np.random.choice(list(range(len(p))), p=p / p.sum())
                #     if (ind, p[ind]) not in words_prob:
                #         words_prob.append((ind, p[ind]))

                p = p.tolist()
                words_prob = [(p.index(word_prob[i]), word_prob[i]) for i in range(2**int(bit_num))]
                nodes = Huffman_Encoding.createNodes([item[1] for item in words_prob])
                root = Huffman_Encoding.createHuffmanTree(nodes)
                codes = Huffman_Encoding.huffmanEncoding(nodes, root)
                for i in range(2**int(bit_num)):
                    if bit_stream[bit_index:bit_index+i+1] in codes:
                        code_index = codes.index(bit_stream[bit_index:bit_index+i+1])
                        gen = words_prob[code_index][0]
                        test_data = np.int32(gen)
                        gen_res.append(idx_to_word[gen])
                        if idx_to_word[gen] in ['\n', '']:
                            break
                        bit += bit_stream[bit_index: bit_index+i+1]
                        bit_index = bit_index+i+1
                        break

            if len(gen_res) < 5:
                continue

            gen_sen = ' '.join([word for word in gen_res if word not in ["\n", ""]])
            count = count + 1
            outfile.write(gen_sen+"\n")
            bitfile.write(bit)


if __name__ == "__main__":
    tf.app.run()
