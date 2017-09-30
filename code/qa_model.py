from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
from datetime import datetime

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.nn import dynamic_rnn, bidirectional_dynamic_rnn

from evaluate import exact_match_score, f1_score
from util import ConfusionMatrix, Progbar, minibatches, get_minibatches
from defs import LBLS

logging.basicConfig(level=logging.INFO)
LOGDIR = '~tensorboard2'


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


def linear(input_, output_size, scope=None):
    '''
    This linear func is for LSTM with attention because original method to calculate linear map is no longer exists in tf v1.0.
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
      Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError(
            "Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable(
            "Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


class LSTMAttnCell(tf.contrib.rnn.LSTMCell):
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        super(LSTMAttnCell, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        lstm_out, lstm_state = super(
            LSTMAttnCell, self).__call__(inputs, state, scope)
        # with tf.device('/gpu:0'):
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn"):
                ht = linear(lstm_out, self._num_units)
                ht = tf.expand_dims(ht, axis=1)
            scores = tf.reduce_sum(
                self.hs * ht, reduction_indices=2, keep_dims=True)
            scores = tf.exp(scores - tf.reduce_max(scores,
                                                   reduction_indices=1, keep_dims=True))
            scores = scores / (1e-6 + tf.reduce_sum(scores,
                                                    reduction_indices=1, keep_dims=True))
            context = tf.reduce_sum(self.hs * scores, reduction_indices=1)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(
                    linear(tf.concat([context, lstm_out], 1), self._num_units))

        return (out, tf.contrib.rnn.LSTMStateTuple(out, out))


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def length(self, mask):
        used = tf.cast(mask, tf.int32)
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def encode_questions(self, inputs, masks, encoder_state_input=None):
        '''Encode questions by Bidirectional LSTM.
        Args:
            inputs: Symbolic representations of your input with shape = (batch_size, length/max_length, embed_size)
            masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
            encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        Returns:
            final_state: The final hidden state of both of forward/backward LSTM.
            states: Concatenated outputs of Bidirectional LSTM.
        '''

        if encoder_state_input == None:
            encoder_state_input = tf.zeros([1, self.size])

        cell_size = self.size
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.size)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.size)

        with tf.variable_scope("bi_LSTM"):
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                dtype=tf.float32,
                sequence_length=self.length(masks),
                inputs=inputs,
                time_major=False
            )

        final_state_fw = final_state[0].h
        final_state_bw = final_state[1].h
        final_state = tf.concat([final_state_fw, final_state_bw], 1)
        states = tf.concat(outputs, 2)
        return final_state, states

    def encode_w_attn(self, inputs, masks, prev_states, scope="", reuse=False):
        '''Encode Documen(paragraphs) by Bidirectional LSTM with attentions.
        Args:
            inputs: Symbolic representations of your input with shape = (batch_size, length/max_length, embed_size)
            masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
            encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
            prev_states: the output which question encoder retuns.

        Returns:
            final_staete: The final hidden state of both of forward/backward LSTM.
            staetes: Concatenated outputs of Bidirectional LSTM.
        '''

        cell_size = self.size
        prev_states_fw, prev_states_bw = tf.split(prev_states, 2, 2)

        attn_cell_fw = LSTMAttnCell(cell_size, prev_states_fw)
        attn_cell_bw = LSTMAttnCell(cell_size, prev_states_bw)
        with vs.variable_scope(scope, reuse):
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                attn_cell_fw,
                attn_cell_bw,
                dtype=tf.float32,
                sequence_length=self.length(masks),
                inputs=inputs,
                time_major=False
            )

        final_state_fw = final_state[0].h
        final_state_bw = final_state[1].h
        final_state = tf.concat([final_state_fw, final_state_bw], 1)
        states = tf.concat(outputs, 2)

        return final_state, states


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size * 2

    def match_LSTM(self, questions_states, paragraph_states, question_length, paragraph_length):
        '''Get word by word attention by MatchLSTM model.
        Args:
            question_stetes: The outputs of the Bidirectional LSTM encoder for questions returns.
                            with shape = (batch_size, question_length, output_size)
            paragraph_states: The outputs of the Bidirectional LSTM encoder with attention for paragraph returns.
                            with shape = (batch_size, paragraph_length, output_size)
            question_length: The (max) length of question, the default is 100.
            paragraph_length: The (max) length of paragraph, the default is 766.
        Returns:
            know_rep: concatanation of forward hidden states and backward hidden states.

        '''
        cell = tf.contrib.rnn.LSTMCell(
            num_units=self.output_size // 2, state_is_tuple=False)
        fw_states = []

        with tf.device('/gpu:0'):
            with tf.variable_scope("Forward_Match-LSTM"):
                W_q = tf.get_variable("W_q", shape=(
                    self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
                W_p = tf.get_variable("W_p", shape=(
                    self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
                W_r = tf.get_variable("W_r", shape=(
                    self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
                b_p = tf.get_variable("b_p", shape=(
                    1, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
                w = tf.get_variable("w", shape=(self.output_size, 1),
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=(
                    1, 1), initializer=tf.contrib.layers.xavier_initializer())
                state = tf.zeros([10, self.output_size])

                # tf.summary.histogram("Forward_Match-LSTM_W_Q", W_q)
                # tf.summary.histogram("Forward_Match-LSTM_W_p", W_p)
                # tf.summary.histogram("Forward_Match-LSTM_W_r", W_r)
                # tf.summary.histogram("Forward_Match-LSTM_biases", b_p)
                # tf.summary.histogram("Forward_Match-LSTM_biases", b)
                # tf.summary.histogram("Forward_Match-LSTM_weights", w)
                # tf.summary.histogram("Forward_Match-LSTM_states", state)

                for time_step in range(paragraph_length):
                    p_state = paragraph_states[:, time_step, :]

                    # Reshapes to 2-D, (batch_size * q_length) *
                    # output_shape(=l)
                    X_ = tf.reshape(questions_states, [-1, self.output_size])

                    # an Intermediate value of question states.
                    # The shape is (Q, l)
                    q_intm = tf.matmul(X_, W_q)

                    # W_pH_p + W_th_{i-1} + b_q
                    # The shape is (batch_size, output_size, 1), R^{l x 1}
                    p_intm = tf.matmul(p_state, W_p) + \
                        tf.matmul(state, W_r) + b_p

                    # The e_q is a column vector filled with 1, whose length=Q
                    # e_q_t is transposed matrix of e_q.
                    # The sape is (Q, batch_size)
                    e_q_t = tf.ones([tf.shape(X_)[0], tf.shape(p_state)[0]])

                    # Produces matrix by repeating vector/matrix p_intm Q
                    # times.
                    p_intm_converted = tf.matmul(e_q_t, p_intm)
                    # p_intm_converted shape is (1000, 200), same as q_intm
                    sum_p_q = q_intm + p_intm_converted

                    # G is the output value of activation function tanh.
                    G = tf.nn.tanh(sum_p_q)

                    # calculate attention weight, for the i th token.
                    # atten shape is now (1000, 1).
                    atten = tf.nn.softmax(tf.matmul(G, w) + b)

                    # Reshapes attention vector, shape is (b_size=10, 1, Q=100)
                    atten = tf.reshape(atten, [-1, 1, question_length])

                    X_ = tf.reshape(questions_states,
                                    [-1, question_length, self.output_size])
                    # After being reshapes, the X_ is now (batch_size=10,
                    # question_length=100, output_size=200)

                    p_z = tf.matmul(atten, X_)

                    p_z = tf.reshape(p_z, [-1, self.output_size])
                    z = tf.concat([p_state, p_z], 1)

                    o, state = cell(z, state)

                    fw_states.append(state)
                    tf.get_variable_scope().reuse_variables()

            fw_states = tf.stack(fw_states)
            fw_states = tf.transpose(fw_states, perm=(1, 0, 2))

            cell = tf.contrib.rnn.LSTMCell(
                num_units=self.output_size // 2, state_is_tuple=False)
            bk_states = []
            print("Forward MatchLSTM done.")

            with tf.variable_scope("Backward_Match-LSTM"):
                W_q = tf.get_variable("W_q", shape=(
                    self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
                W_p = tf.get_variable("W_p", shape=(
                    self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
                W_r = tf.get_variable("W_r", shape=(
                    self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
                b_p = tf.get_variable("b_p", shape=(
                    1, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
                w = tf.get_variable("w", shape=(self.output_size, 1),
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=(
                    1, 1), initializer=tf.contrib.layers.xavier_initializer())
                state = tf.zeros([10, self.output_size])

                # tf.summary.histogram("Backword_Match-LSTM_W_Q", W_q)
                # tf.summary.histogram("Backword_Match-LSTM_W_p", W_p)
                # tf.summary.histogram("Backword_Match-LSTM_W_r", W_r)
                # tf.summary.histogram("Backword_Match-LSTM_biases", b_p)
                # tf.summary.histogram("Backword_Match-LSTM_biases", b)
                # tf.summary.histogram("Backword_Match-LSTM_weights", w)
                # tf.summary.histogram("Backword_Match-LSTM_states", state)

                for time_step in range(paragraph_length):
                    p_state = paragraph_states[:, time_step, :]
                    X_ = tf.reshape(questions_states, [-1, self.output_size])

                    e_q_t = tf.ones([tf.shape(X_)[0], tf.shape(p_state)[0]])

                    q_intm = tf.matmul(X_, W_q)
                    p_intm = tf.matmul(p_state, W_p) + \
                        tf.matmul(state, W_r) + b_p

                    p_intm_converted = tf.matmul(e_q_t, p_intm)
                    sum_p_q = q_intm + p_intm_converted
                    G = tf.nn.tanh(sum_p_q)

                    atten = tf.nn.softmax(tf.matmul(G, w) + b)
                    atten = tf.reshape(atten, [-1, 1, question_length])
                    X_ = tf.reshape(questions_states,
                                    [-1, question_length, self.output_size])

                    p_z = tf.matmul(atten, X_)

                    p_z = tf.reshape(p_z, [-1, self.output_size])
                    z = tf.concat([p_state, p_z], 1)
                    o, state = cell(z, state)

                    bk_states.append(state)
                    tf.get_variable_scope().reuse_variables()
            print("backward decoding done.")

            bk_states = tf.stack(bk_states)
            bk_states = tf.transpose(bk_states, perm=(1, 0, 2))

        knowledge_rep = tf.concat([fw_states, bk_states], 2)

        print("The shape of know_rep {0}".format(knowledge_rep.get_shape()))

        return knowledge_rep

    def decode(self, knowledge_rep, paragraph_length):
        """Decode the knowledge representation and calculate a probability estimation
        over all paragraph tokens.
        Args:
            knowledge_rep: The cancatnation of forward/backward hidden states retuned by MatchLSTM().
            paragraph_length = The max length of paragraph.
        Returns:
            beta_s: Attention weight vector for the start position of an answer span.
            beta_e: Attention weight vector for the end position of an answer span.

        """
        # TODO: Add zero vector at 0 as the original PointerNet.
        output_size = self.output_size

        cell = tf.contrib.rnn.LSTMCell(
            num_units=output_size // 2, state_is_tuple=False)
        beta_s = []

        with tf.device('/gpu:0'):
            with tf.variable_scope("Boundary-LSTM_start"):
                V = tf.get_variable("V", shape=(
                    2 * output_size, output_size), initializer=tf.contrib.layers.xavier_initializer())
                b_a = tf.get_variable("b_a", shape=(
                    1, output_size), initializer=tf.contrib.layers.xavier_initializer())
                W_a = tf.get_variable("W_a", shape=(
                    output_size, output_size), initializer=tf.contrib.layers.xavier_initializer())
                c = tf.get_variable("c", shape=(
                    1, 1), initializer=tf.contrib.layers.xavier_initializer())
                v = tf.get_variable("v", shape=(output_size, 1),
                                    initializer=tf.contrib.layers.xavier_initializer())
                state = tf.zeros([10, output_size])

                # tf.summary.histogram("Boundary-LSTM_start_V", V)
                # tf.summary.histogram("Boundary-LSTM_start_b_a", b_a)
                # tf.summary.histogram("Boundary-LSTM_start_W_a", W_a)
                # tf.summary.histogram("Boundary-LSTM_start_c", c)
                # tf.summary.histogram("Boundary-LSTM_start_v", v)
                # tf.summary.histogram("Boundary-LSTM_start_states", state)

                for time_step in range(paragraph_length):
                    H_r = tf.reshape(knowledge_rep, [-1, 2 * output_size])
                    H_r_0 = tf.concat(H_r, tf.zeros([1, 2 * output_size], tf.float32))

                    first = tf.matmul(H_r_0, V)
                    e_p1_t = tf.ones(
                        [tf.shape(H_r_0)[0], tf.shape(knowledge_rep)[0]])
                    second = tf.matmul(e_p1_t, tf.matmul(state, W_a) + b_a)
                    F_s = tf.nn.tanh(first + second)

                    beta = tf.nn.softmax(tf.matmul(F_s, v) + c)
                    probab_s = tf.reshape(beta, shape=[-1, paragraph_length])
                    beta_s.append(probab_s)
                    # prob shape (10, 766)
                    beta = tf.reshape(beta, [-1, 1, paragraph_length])
                    H_r = tf.reshape(
                        knowledge_rep, [-1, paragraph_length, 2 * output_size])

                    z = tf.matmul(beta, H_r)
                    z = tf.reshape(z, [-1, 2 * output_size])
                    _, state = cell(z, state, scope="Boundary-LSTM_start")
                    tf.get_variable_scope().reuse_variables()

            beta_s = tf.stack(beta_s)
            beta_s = tf.transpose(beta_s, perm=(1, 0, 2))

            beta_e = []
            with tf.variable_scope("Boundary-LSTM_end"):
                cell = tf.contrib.rnn.LSTMCell(
                    num_units=output_size // 2, state_is_tuple=False)
                V = tf.get_variable("V", shape=(
                    2 * output_size, output_size), initializer=tf.contrib.layers.xavier_initializer())
                b_a = tf.get_variable("b_a", shape=(
                    1, output_size), initializer=tf.contrib.layers.xavier_initializer())
                W_a = tf.get_variable("W_a", shape=(
                    output_size, output_size), initializer=tf.contrib.layers.xavier_initializer())
                c = tf.get_variable("c", shape=(
                    1, 1), initializer=tf.contrib.layers.xavier_initializer())
                v = tf.get_variable("v", shape=(output_size, 1),
                                    initializer=tf.contrib.layers.xavier_initializer())

                # tf.summary.histogram("Boundary-LSTM_end_V", V)
                # tf.summary.histogram("Boundary-LSTM_end_b_a", b_a)
                # tf.summary.histogram("Boundary-LSTM_end_W_a", W_a)
                # tf.summary.histogram("Boundary-LSTM_end_c", c)
                # tf.summary.histogram("Boundary-LSTM_end_v", v)
                # tf.summary.histogram("Boundary-LSTM_end_states", state)
                # 10 is output size.
                state = tf.zeros([10, output_size])
                for time_step in range(paragraph_length):
                    H_r = tf.reshape(knowledge_rep, [-1, 2 * output_size])
                    H_r_0 = tf.concat(H_r, tf.zeros([1, 2 * output_size], tf.float32))
                    first = tf.matmul(H_r_0, V)

                    e_p1_t = tf.ones(
                        [tf.shape(H_r_0)[0], tf.shape(knowledge_rep)[0]])
                    second = tf.matmul(e_p1_t, tf.matmul(state, W_a) + b_a)
                    F_s = tf.nn.tanh(first + second)

                    beta = tf.nn.softmax(tf.matmul(F_s, v) + c)

                    probab_e = tf.reshape(beta, shape=[-1, paragraph_length])
                    beta_e.append(probab_e)

                    beta = tf.reshape(beta, [-1, 1, paragraph_length])
                    H_r = tf.reshape(
                        knowledge_rep, [-1, paragraph_length, 2 * output_size])
                    z = tf.matmul(beta, H_r)
                    z = tf.reshape(z, [-1, 2 * output_size])

                    _, state = cell(z, state, scope="Boundary-LSTM_start")
                    tf.get_variable_scope().reuse_variables()
            beta_e = tf.stack(beta_e)
            beta_e = tf.transpose(beta_e, perm=(1, 0, 2))

        return beta_s, beta_e


class QASystem(object):
    def __init__(self, encoder, decoder, args, pretrained_embeddings):
        """
        Initializes your System
        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.config = args
        self.pretrained_embeddings = pretrained_embeddings
        # ==== set up placeholder tokens ========
        self.p_max_length = self.config.paragraph_size
        self.embed_size = encoder.vocab_dim
        self.q_max_length = self.config.question_size
        self.q_placeholder = tf.placeholder(
            tf.int32, (None, self.q_max_length))
        self.p_placeholder = tf.placeholder(
            tf.int32, (None, self.p_max_length))
        self.start_labels_placeholder = tf.placeholder(
            tf.int32, (None, self.p_max_length))
        self.end_labels_placeholder = tf.placeholder(
            tf.int32, (None, self.p_max_length))
        self.q_mask_placeholder = tf.placeholder(
            tf.bool, (None, self.q_max_length))
        self.p_mask_placeholder = tf.placeholder(
            tf.bool, (None, self.p_max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, ())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.start_index_loss, self.end_index_loss = self.setup_loss()

        # ==== set up training/updating procedure ====
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(
            self.config.learning_rate, step, 1, self.config.learning_rate_decay)

        optimizer = get_optimizer(self.config.optimizer)(rate)

        # Should I use apply+gradient() or minimize()?
        s_grads, s_vars = zip(
            *optimizer.compute_gradients(self.start_index_loss))
        e_grads, e_vars = zip(
            *optimizer.compute_gradients(self.end_index_loss))

        if self.config.max_gradient_norm:
            c_s_grads, self.s_grad_norm = tf.clip_by_global_norm(
                s_grads, self.config.max_gradient_norm)
            c_e_grads, self.e_grad_norm = tf.clip_by_global_norm(
                e_grads, self.config.max_gradient_norm)
        else:
            c_s_grads = s_grads
            c_e_grads = e_grads
            self.s_grad_norm = tf.global_norm(s_grads)
            self.e_grad_norm = tf.global_norm(e_grads)

        # TODO: A bug where the losses are not updated at all....
        # self.start_index_train_op = optimizer.apply_gradients(
        #     zip(c_s_grads, s_vars))
        # self.end_index_train_op = optimizer.apply_gradients(
        #     zip(c_e_grads, e_vars))
        self.start_index_train_op = optimizer.minimize(
            self.start_index_loss)
        self.end_index_train_op = optimizer.minimize(self.end_index_loss)

    def setup_system(self):
        """Set up the entire layers, LSTMPreprocessing Layer, MatchLSTM Layer, AnswerPointer Layer(Boundary model.)
        First, it encode questions and paragraph separately, and then decode them with MatchLSTM(match_LSTM()).
        Lastly, it calculate the answer possibility prediction over paragraph words by PointerNet(decode()).
        """
        encoded_q, self.q_states = self.encoder.encode_questions(
            self.q_embeddings, self.q_mask_placeholder, None)
        encoded_p, self.p_states = self.encoder.encode_w_attn(
            self.p_embeddings, self.p_mask_placeholder, self.q_states, scope="", reuse=False)

        self.knowledge_rep = self.decoder.match_LSTM(
            self.q_states, self.p_states, self.q_max_length, self.p_max_length)
        self.preds = self.decoder.decode(
            self.knowledge_rep, self.p_max_length)

    def setup_loss(self):
        """Sets up losses based on decoder.decode inputs, refered as self.preds.
        *preds[0] -> start index predictons, preds[1] -> end index predictions.

        Returns:
            start_index_loss: The op to calculate the loss for the start index prediction.
            end_index_loss: The op to calculate the loss for the end index prediction.
        """
        preds = np.array(self.preds)
        with vs.variable_scope("start_index_loss"):
            loss_tensor = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.preds[0], labels=self.start_labels_placeholder), self.p_mask_placeholder)
            start_index_loss = tf.reduce_mean(loss_tensor, 0)
            tf.summary.scalar(
                'start_index_cross_entroy_loss', start_index_loss)

        with vs.variable_scope("end_index_loss"):
            loss_tensor = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=preds[1], labels=self.end_labels_placeholder), self.p_mask_placeholder)
            end_index_loss = tf.reduce_mean(loss_tensor, 0)
            tf.summary.scalar(
                'end_index_cross_entroy_loss', end_index_loss)

        return start_index_loss, end_index_loss

    def setup_embeddings(self):
        """Loads distributed word representations based on placeholder tokens
        Also adds some syntactic information such as postag, ner and exact match in the future.
        """
        with vs.variable_scope("embeddings"):
            self.pretrained_embeddings = tf.Variable(
                self.pretrained_embeddings, trainable=False, dtype=tf.float32)
            q_embeddings = tf.nn.embedding_lookup(
                self.pretrained_embeddings, self.q_placeholder)
            self.q_embeddings = tf.reshape(
                q_embeddings, shape=[-1, self.config.question_size, 1 * self.embed_size])
            p_embeddings = tf.nn.embedding_lookup(
                self.pretrained_embeddings, self.p_placeholder)
            self.p_embeddings = tf.reshape(
                p_embeddings, shape=[-1, self.config.paragraph_size, 1 * self.embed_size])

    def optimize(self, session, question_batch, context_batch, labels_batch=None, q_mask_batch=None, p_mask_batch=None):
        """Optimize the model.
        Args:
            Session:
            question_batch:
            context_batch:
            labels_batch:
            q_mask_batch:
            p_mask_batch:
        Returns:
            start_index_loss: The loss on start index of answer span predictions.
            end_index_loss: The loss on end index of answer span predictions.
        """
        input_feed = self.create_feed_dict(
            question_batch, context_batch, labels_batch, q_mask_batch, p_mask_batch)

        start_output_feed = [self.start_index_train_op, self.start_index_loss]
        _, start_index_loss_val = session.run(start_output_feed, input_feed)
        end_output_feed = [self.end_index_train_op, self.end_index_loss]
        _, end_index_loss_val = session.run(end_output_feed, input_feed)
        print("start index loss : " + str(start_index_loss))
        print("end index loss : " + str(end_index_loss))

        return start_index_loss_val, end_index_loss_val

    def test(self, session, valid_x, valid_y):
        # TODO: Add test.
        input_feed = {}

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, train_x, mask):
        # TODO: Add more descriptions.
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = self.create_feed_dict(train_x['Questions'], context_batch['Paragraphs'], train_x['Labels'], train_x['Questions_masks'], train_x['Paragraphs_masks'])

        if train_x is not None:
            input_feed[self.q_placeholder] = train_x['Questions']
            input_feed[self.p_placeholder] = train_x['Paragraphs']
        if mask is not None:
            input_feed[self.q_mask_placeholder] = train_x['Questions_masks']
            input_feed[self.p_mask_placeholder] = train_x['Paragraphs_masks']

        output_feed = [self.preds]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def create_feed_dict(self, question_batch, context_batch, labels_batch=None, q_mask_batch=None, p_mask_batch=None):
        """Creates the feed_dict for the model.
        Args:
            question_batch: The questions in each minibatch, represented by (int) wordID's sequence.
            context_batch: The paragraphs in each minibatch, represented by (int) wordID's sequence.
            labels_batch: The answer labels in each minibatch, represented by [start_index, end_index]
            q_mask_batch: The questions masks in each minibatch, represented by 11111000000.... sequence.
            p_mask_batch: The paragraohs masks in each minibatch, represented by 11111000000.... sequence.

        Returns:
            feed_dict: The dictioary of input data for the minibatch.
        """
        feed_dict = {}

        feed_dict[self.q_placeholder] = question_batch
        feed_dict[self.p_placeholder] = context_batch
        feed_dict[self.q_mask_placeholder] = q_mask_batch
        feed_dict[self.p_mask_placeholder] = p_mask_batch

        if labels_batch is not None:
            start_index = [labels[0] for labels in labels_batch]
            end_index = [labels[1] for labels in labels_batch]
            start_labels = []
            end_labels = []
            for i in range(len(labels_batch)):
                start_label_question = [
                    1 if j == start_index[i] else 0 for j in range(self.p_max_length)]
                end_label_question = [1 if j == end_index[i]
                                      else 0 for j in range(self.p_max_length)]
                start_labels.append(start_label_question)
                end_labels.append(end_label_question)

        feed_dict[self.start_labels_placeholder] = start_labels
        feed_dict[self.end_labels_placeholder] = end_labels

        return feed_dict

    def run_epoch(self, session, inputs):
        """Runs an epoch of training.
        Args:
            sess: tf.Session() object
            inputs: datasets represented as a dictionary
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        losses = []
        n_minibatches, total_loss = 0, 0

        for [question_batch, context_batch, labels_batch, q_mask_batch, p_mask_batch] in \
                get_minibatches([inputs['Questions'], inputs['Paragraphs'], inputs['Labels'], inputs['Questions_masks'], inputs['Paragraphs_masks']], self.config.batch_size):
            start_index_loss_val, end_index_loss_val = self.optimize(
                session, question_batch, context_batch, labels_batch, q_mask_batch, p_mask_batch)
            n_minibatches += 1

            losses.append([start_index_loss_val, end_index_loss_val])

        mean = np.mean(losses, axis=0)
        logging.info(
            "Logged mean epoch losses: start : %f end : %f ", mean[0], mean[1])

        return losses

    def answer(self, session, test_x, mask):
        yp, yp2 = self.decode(session, test_x, mask)
        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)
        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
            valid_cost = self.test(sess, valid_x, valid_y)

        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        idx_sample = np.random.randint(
            0, dataset['Questions'].shape[0], sample)

        examples = {}
        examples['Questions'] = dataset['Questions'][idx_sample]
        examples['Paragraphs'] = dataset['Paragraphs'][idx_sample]
        examples['Questions_masks'] = dataset['Questions'][idx_sample]
        examples['Paragraphs_masks'] = dataset['Paragraphs'][idx_sample]
        examples['Labels'] = dataset['Labels'][idx_sample]

        correct_preds, total_correct, total_preds = 0., 0., 0.
        masks = True
        for _, labels, labels_ in self.answer(session, examples, masks):
            pred = set()
            if labels_[0] <= labels_[1]:
                pred = set(range(labels_[0], labels_[1] + 1))
            gold = set(range(labels[0], labels[1] + 1))

            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        em = correct_preds
        tf.summary.scalar("f1", f1)
        tf.summary.scalar("em", em)

        merged = tf.summary.merge_all()
        summary = sess.run(merged, feed_dict=feed_dict(False))

        if log:
            logging.info(
                "F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em, summary

    def train(self, session, dataset, train_dir):
        """Train the models. Run the epochs and evaluate the score.
        Args:
            session: tf.Session() object.
            dateset: Input dateset created by code/qa_data.py. The dataset includes
                    "Questions", "Questions_masks", "Paragraphs", "Paragraphs_masks", "Labels".
            train_dir: The directory name which stores the training results.
        Returns:
            best_score: The best F1 score during the training.
        """
        results_path = os.path.join(
            train_dir, "{:%Y%m%d_%H%M%S}".format(datetime.now()))
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(
            tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" %
                     (num_params, toc - tic))
        best_score = 0.
        print("Questions_masks")

        train_writer = tf.summary.FileWriter(LOGDIR + 'train')
        train_writer.add_graph(session.graph)
        saver = tf.train.Saver()

        for epoch in range(self.config.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.config.epochs)
            logging.info("Best score so far: " + str(best_score))
            loss = self.run_epoch(session, dataset)

            f1, em, s = self.evaluate_answer(
                session, dataset, sample=800, log=True)

            train_writer.add_summary(s, epoch)
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), epoch)
            logging.info("loss: " + str(start_loss) + " f1: " +
                         str(f1) + " em:" + str(em))
            if f1 > best_score:
                best_score = f1
                logging.info(
                    "New best score! Saving model in %s", results_path)
                if self.saver:
                    self.saver.save(session, results_path)
            print("")

        return best_score
