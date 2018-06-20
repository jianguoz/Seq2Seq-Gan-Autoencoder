import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import time


###========================== prepare data===========================================###

# preprocessed data
# from datasets.cornell_corpus import data
# import data_utils
#
# # load data from pickle and npy files
# metadata, idx_q, idx_a = data.load_data(PATH='datasets/cornell_corpus/') # PATH='datasets/cornell_corpus/'
# (trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)
from data.twitter import data
metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')                   # Twitter
# from data.cornell_corpus import data
# metadata, idx_q, idx_a = data.load_data(PATH='data/cornell_corpus/')          # Cornell Moive
(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

trainX = trainX.tolist()
trainY = trainY.tolist()
testX = testX.tolist()
testY = testY.tolist()
validX = validX.tolist()
validY = validY.tolist()

#print(trainX[1])

trainX = tl.prepro.remove_pad_sequences(trainX)
trainY = tl.prepro.remove_pad_sequences(trainY)
testX = tl.prepro.remove_pad_sequences(testX)
testY = tl.prepro.remove_pad_sequences(testY)
validX = tl.prepro.remove_pad_sequences(validX)
validY = tl.prepro.remove_pad_sequences(validY)

# parameters
xseq_len = len(trainX)
yseq_len = len(trainY)
assert xseq_len == yseq_len

batch_size = 32
n_step = int(xseq_len/batch_size)
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

w2idx = metadata['w2idx']   # dict  word 2 index
idx2w = metadata['idx2w']   # list index 2 word

unk_id = w2idx['unk']   # 1
pad_id = w2idx['_']     # 0

start_id = xvocab_size  # 8002
end_id = xvocab_size+1  # 8003

w2idx.update({'start_id': start_id})
w2idx.update({'end_id': end_id})
idx2w = idx2w + ['start_id', 'end_id']

xvocab_size = yvocab_size = xvocab_size + 2


""" A data for Seq2Seq should look like this:
input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
target_mask : [1, 1, 1, 1, 0]
"""

# show trainX[10] as an example
print("encode_seqs", [idx2w[id] for id in trainX[10]])
target_seqs = tl.prepro.sequences_add_end_id([trainY[10]], end_id=end_id)[0]
    # target_seqs = tl.prepro.remove_pad_sequences([target_seqs], pad_id=pad_id)[0]
print("target_seqs", [idx2w[id] for id in target_seqs])
decode_seqs = tl.prepro.sequences_add_start_id([trainY[10]], start_id=start_id, remove_last=False)[0]
    # decode_seqs = tl.prepro.remove_pad_sequences([decode_seqs], pad_id=pad_id)[0]
print("decode_seqs", [idx2w[id] for id in decode_seqs])
target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]
print("target_mask", target_mask)
print(len(target_seqs), len(decode_seqs), len(target_mask))

###========================== prepare data===========================================###


###========================== main model===========================================###
def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size=xvocab_size,
                embedding_size=emb_dim,
                name = 'seq_embedding'
            )
            vs.reuse_variables()
            tl.layers.set_name_reuse() #remove if TL version == 1.8.0+

            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size=xvocab_size,
                embedding_size=emb_dim,
                name = 'seq_embedding'
            )
        net_rnn = Seq2Seq(net_encode, net_decode,
                          cell_fn=tf.contrib.rnn.BasicLSTMCell,
                          n_hidden=emb_dim,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length=retrieve_seq_length_op2(encode_seqs),
                          decode_sequence_length=retrieve_seq_length_op2(decode_seqs),
                          initial_state_encode=None,
                          initial_state_decode=None,
                          dropout=(0.5 if is_train else None),
                          n_layer=3,
                          return_seq_2d=True,
                          name='Seq2Seq')
        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='out')
    return net_out, net_rnn

# model for training
encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='encode_seqs')
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='decode_seqs')
target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='target_seqs')
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='target_mask')
net_out, _ = model(encode_seqs, decode_seqs, is_train=True, reuse=False)

# model for inference
encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name='encode_seqs_2')
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name='decode_seqs_2')
net_infer, net_rnn = model(encode_seqs2, decode_seqs2, is_train=False, reuse=True)
y = tf.nn.softmax(net_infer.outputs)

# loss for training
loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, input_mask=target_mask,
                                          return_details=False, name='cost')
net_out.print_params(False)

lr=0.0001
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# session
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name='n.npa', network=net_infer)

# =====train=========
n_epochs = 50
for epoch in range(n_epochs):
    epoch_time = time.time()
    # shuffle training data
    from sklearn.utils import shuffle
    trainX, trainY = shuffle(trainX, trainY)
    # train an epoch
    total_err, n_iter = 0, 0 # total_error, #iterations
    for X, Y in tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False):
        step_time = time.time()

        X = tl.prepro.pad_sequences(X)
        _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
        _target_seqs = tl.prepro.pad_sequences(_target_seqs)

        _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
        _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
        _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

        _, err = sess.run([train_op, loss],
                          feed_dict={encode_seqs: X,
                                     decode_seqs: _decode_seqs,
                                     target_seqs: _target_seqs,
                                     target_mask: _target_mask}
                          )

        if n_iter % 200 == 0:
            print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, n_epochs, n_iter, n_step, err, time.time() - step_time))
        total_err += err
        n_iter += 1

        # ====inference===
        if n_iter % 1000 == 0:
            seeds = ["happy birthday have a nice day",
                     "so fun to see what the special effects team created for",
                     "donald trump won last nights presidential debate according to snap online polls"]
            for seed in seeds:
                print('Query >', seed)
                seed_id = [w2idx[w] for w in seed.split(" ")]
                for _ in range(5): # 1 Query --> 5 reply
                    # 1. encode, get state
                    state = sess.run(net_rnn.final_state_encode,
                                    feed_dict={encode_seqs2: [seed_id]}
                                    )
                    # 2. decode, feed start_id, get first word
                    o, state = sess.run([y, net_rnn.final_state_decode],
                                        feed_dict={net_rnn.initial_state_decode: state,
                                                   decode_seqs2: [[start_id]]}
                                        )
                    w_id = tl.nlp.sample_top(o[0], top_k=3)
                    w = idx2w[w_id]
                    # 3. decode, feed state iteratively
                    sentence = [w]
                    for _ in range(30): # max_sequence_length = 30
                        o, state = sess.run([y, net_rnn.final_state_decode],
                                            feed_dict={net_rnn.initial_state_decode:state,
                                                       decode_seqs2: [[w_id]]}
                                            )
                        w_id = tl.nlp.sample_top(o[0], top_k=2)
                        w = idx2w[w_id]
                        if w_id == end_id:
                            break
                        sentence = sentence + [w]
                    print('>', ' '.join(sentence))

    print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % (epoch, n_epochs, total_err / n_iter, time.time() - epoch_time))
    tl.files.save_npz(net_infer.all_params, name='n.npz', sess=sess)
