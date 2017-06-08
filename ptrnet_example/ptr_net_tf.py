import tensorflow as tf
#from tensorflow.contrib.layers import xavier_initializer as xinit

from recurrence import *
from train import *

from tqdm import tqdm

class PointerNet():

    def __init__(self, L=10, B=32, num_indices=2, 
                hdim=10, lr=.9):

        self.L = L
        self.B = B
        self.num_indices = num_indices
        self.hdim = hdim


        tf.reset_default_graph()

        self.init = tf.random_normal_initializer(0.0, 0.5)

        self.inputs = tf.placeholder(tf.float32, name='inputs', shape=[B,L])
        self.targets = tf.placeholder(tf.float32, name='targets', 
                                         shape=[num_indices, B, L])

        with tf.variable_scope('encoder'):
            # encoder -> encoder states
            estates = self.encoder(self.inputs)

        with tf.variable_scope('decoder'):
            # decoder -> decoder outputs, logits
            dec_outputs, probs = self.pointer_decoder(estates, as_prob=True) 

        '''
            we are forsaking cross entropy for now

        # cross entropy
        ce = tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits, labels=self.targets, 
                                name='cross_entropy')
        '''

        # self.loss = tf.sqrt(tf.reduce_mean(tf.pow(probs - self.targets, 2.0)))

        self.loss = tf.losses.mean_squared_error(labels=self.targets, predictions=probs)

        # optimization
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss)

        # init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def encoder(self, inputs):
        ecell = gru(self.hdim)
        enc_init_state = ecell.zero_state(self.B, tf.float32)
        estates = [enc_init_state]
        for i in range(self.L):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            input_ = inputs[:, i:i+1]

            # project input [B, 1] -> [B, hdim]
            We = tf.get_variable('W_e', [1, self.hdim], initializer=self.init)
            be = tf.get_variable('b_e', [self.hdim], initializer=self.init)
            input_ = tf.nn.elu(tf.matmul(input_, We) + be, name='cell_input')

            # step
            output, state = ecell(input_, estates[-1])
            estates.append(state)

        return estates[1:]

    def pointer_decoder(self, estates, as_prob=False):
        # special generation symbol
        special_sym_value = 20.
        special_sym = tf.constant(special_sym_value, shape=[self.B,1], dtype=tf.float32)
        # decoder states
        dec_init_state = estates[-1]
        dstates = [dec_init_state]
        # decoder input
        d_input_ = special_sym

        # create cell
        dcell = gru(self.hdim)

        logits = []
        probs = []
        dec_outputs = []
        for i in range(self.num_indices):
            if i>0:
                tf.get_variable_scope().reuse_variables()
                
            # project input
            Wp = tf.get_variable('W_p', [1, self.hdim], initializer=self.init)
            bp = tf.get_variable('b_p', [self.hdim], initializer=self.init)
            
            d_input_ = tf.nn.elu(tf.matmul(d_input_, Wp) + bp, name='decoder_cell_input')
            
            # step
            output, dec_state = dcell(d_input_, dstates[-1])
            
            # project enc/dec states
            W1 = tf.get_variable('W_1', [self.hdim, self.hdim], initializer=self.init)
            W2 = tf.get_variable('W_2', [self.hdim, self.hdim], initializer=self.init)
            ptr_bias = tf.get_variable('ptr_bias', [self.hdim], initializer=self.init)
            v = tf.get_variable('v', [self.hdim, 1], initializer=self.init)
            
            scores = ptr_attention(estates, dec_state,
                          params = {'Wa' : W1, 'Ua' : W2, 'Va' : v},
                          d = self.hdim, timesteps=self.L)
            
            prob_dist = tf.nn.softmax(scores)
            idx = tf.argmax(prob_dist, axis=1)
            
            # get input at index "idx"
            dec_output_i = self.batch_gather_nd(self.inputs, idx)
            
            # output at i is input to i+1
            d_input_ = tf.expand_dims(dec_output_i, axis=-1)
            
            logits.append(scores)
            probs.append(prob_dist)

            dec_outputs.append(dec_output_i)
        
        if as_prob:
            return dec_outputs, tf.stack(probs)

        return dec_outputs, tf.stack(logits)


    def batch_gather_nd(self, t, idx):
        idx = tf.cast(idx, dtype=tf.int32, name='idx_int32')
        range_ = tf.range(start=0, limit=tf.shape(idx)[0])
        idx_pair = tf.stack([range_, idx], axis=1)
        return tf.gather_nd(t, idx_pair)


    def train(self, epochs, num_batches,  batch_size, reset_params):

        # fetch trainset
        trainset = generate_trainset(num_batches=num_batches, 
                        batch_size=batch_size, maxlen=self.L) 

        max_retries = 20
        for k in range(max_retries):

            self.sess.run(tf.global_variables_initializer())
            tf.set_random_seed(1)

            for i in range(epochs):

                avg_loss = 0
                for j in tqdm(range(num_batches)):
                    _, l = self.sess.run([self.train_op, self.loss], feed_dict = {
                                self.inputs : trainset[j][0],
                                self.targets : trainset[j][1]
                            })
                    avg_loss += l

                tqdm.write('{} : {}'.format(i, avg_loss/num_batches))

                if  i>reset_params['steps'] and avg_loss > reset_params['loss']:
                    break

if __name__ == '__main__':
    batch_size = 1
    reset_params = {"steps": 30, "loss": .11}

    ptrnet = PointerNet(L=60, B=batch_size) 
    ptrnet.train(epochs=4000, num_batches= 1024, batch_size=batch_size,
                    reset_params=reset_params)
