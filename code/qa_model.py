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


def preprocess_sequence_data(self, dataset):
    max_c = self.config.max_c
    max_q = self.config.max_q

    stop = next((idx for idx, xi in enumerate(dataset)
                 if len(xi[0]) > max_c), len(dataset))
    assert len(dataset[stop - 1][0]) <= max_c

    c_ids = np.array([xi[0] + [0] * (max_c - len(xi[0]))
                      for xi in dataset[:stop]], dtype=np.int32)
    q_ids = np.array([xi[1] + [0] * (max_q - len(xi[1]))
                      for xi in dataset[:stop]], dtype=np.int32)

    span = np.array([xi[2] for xi in dataset[:stop]], dtype=np.int32)

    c_len = np.array([len(xi[0]) for xi in dataset[:stop]], dtype=np.int32)
    q_len = np.array([len(xi[1]) for xi in dataset[:stop]], dtype=np.int32)

    data_size = c_ids.shape[0]

    assert q_ids.shape[0] == data_size
    assert c_ids.shape == (data_size, max_c)
    assert q_len.shape == (data_size,)
    assert c_len.shape == (data_size,)
    assert span.shape == (data_size, 2)

    return [c_ids, c_len, q_ids, q_len, span]


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

    def encode_questions(self, inputs, masks, encoder_state_input):
        """
        This is an encoder for question. Runing biLSTM over question.
        """
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.
        :param inputs: Symbolic representations of your input with shape = (batch_size, length/max_length, embed_size)
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        if encoder_state_input == None:
            encoder_state_input = tf.zeros([1, self.size])

        # Run bidirectional lstm to encode question.
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
        """
        Run a BiLSTM over the context paragraph conditioned on the question representation.
        """

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
        print("This is encoded paragraph.")
        print(states)
        return final_state, states


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = 2 * output_size

    def match_LSTM(self, questions_states, paragraph_states, question_length, paragraph_length):
        '''
        This is matchLSTM for decode.
        '''

        cell = tf.contrib.rnn.LSTMCell(
            num_units=self.output_size // 2, state_is_tuple=False)
        fw_states = []

        print("The paragraph_states {0}".format(paragraph_states))
        print("The shape of paragraph_states {0}".format(
            paragraph_states.get_shape()))

        with tf.variable_scope("Forward_Match-LSTM"):
            W_q = tf.get_variable("W_q", shape=(
                self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
            W_p = tf.get_variable("W_p", shape=(
                self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
            W_r = tf.get_variable("W_r", shape=(
                self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
            b_p = tf.get_variable("b_p", shape=(
                10, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
            w = tf.get_variable("w", shape=(self.output_size, 1),
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=(
                1, 1), initializer=tf.contrib.layers.xavier_initializer())
            state = tf.zeros([1, self.output_size])

            for time_step in range(paragraph_length):
                p_state = paragraph_states[:, time_step, :]
                X_ = tf.reshape(questions_states, [-1, self.output_size])

                # the dimention of e_Q_t should be [X_shape_X, batch-size]
                e_q_t = tf.ones([tf.shape(X_)[0], tf.shape(p_state)[0]])

                # An intermediate value, converted linearly of question's
                # hidden states.
                q_intm = tf.matmul(X_, W_q)

                # intermediate valuen, W_pH_p + W_th_{i-1} + b_q, the shape
                # should be (length_of hidden state l,1)
                p_intm = tf.matmul(p_state, W_p) + tf.matmul(state, W_r) + b_p

                # expand (l, 1) vector to (l, q), by repeating p_intm to left
                # column. Implemented by outer product of p_intm and  [1 1 1
                # ... 1](length = q)
                p_intm_converted = tf.matmul(e_q_t, p_intm)
                # p_intm_converted shape is (1000, 200), which is the same as
                # q_intm, WqHq
                sum_p_q = q_intm + p_intm_converted

                G = tf.nn.tanh(sum_p_q)
                # G is the output value of activation function tanh.

                atten = tf.nn.softmax(tf.matmul(G, w) + b)
                # calculate attention weight, for the i th token.
                # atten shape is now (1000, 1).
                atten = tf.reshape(atten, [-1, 1, question_length])
                # After being reshaped, the atten shape should be (10, 1, 100)

                X_ = tf.reshape(questions_states,
                                [-1, question_length, self.output_size])
                # After being reshapes, the X_ is now (batch_size=10,
                # question_length=100, output_size=200)

                # TODO: Need to take transpose matrix.
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
        print("Forward decoding done.")

        with tf.variable_scope("Backward_Match-LSTM"):
            W_q = tf.get_variable("W_q", shape=(
                self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
            W_p = tf.get_variable("W_p", shape=(
                self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
            W_r = tf.get_variable("W_r", shape=(
                self.output_size, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
            b_p = tf.get_variable("b_p", shape=(
                10, self.output_size), initializer=tf.contrib.layers.xavier_initializer())
            w = tf.get_variable("w", shape=(self.output_size, 1),
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=(
                1, 1), initializer=tf.contrib.layers.xavier_initializer())
            state = tf.zeros([1, self.output_size])

            for time_step in range(paragraph_length):
                p_state = paragraph_states[:, time_step, :]
                X_ = tf.reshape(questions_states, [-1, self.output_size])

                e_q_t = tf.ones([tf.shape(X_)[0], tf.shape(p_state)[0]])

                # An intermediate value, converted linearly of question's
                # hidden states.
                q_intm = tf.matmul(X_, W_q)
                # intermediate valuen, W_pH_p + W_th_{i-1} + b_q, the shape
                # should be (length_of hidden state l,1)
                p_intm = tf.matmul(p_state, W_p) + tf.matmul(state, W_r) + b_p

                # expand (l, 1) vector to (l, q), by repeating p_intm to left
                # column. Implemented by outer product of p_intm and  [1 1 1
                # ... 1](length = q)
                p_intm_converted = tf.matmul(e_q_t, p_intm)

                sum_p_q = q_intm + p_intm_converted

                # G = tf.nn.tanh(tf.matmul(X_, W_q) + tf.matmul(p_state,
                # W_r) + tf.matmul(state, W_r) + b_p)  # batch_size*Q,l
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

        return knowledge_rep

    def decode(self, knowledge_rep, paragraph_length):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.
        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        output_size = self.output_size

        cell = tf.contrib.rnn.LSTMCell(
            num_units=output_size // 2, state_is_tuple=False)
        beta_s = []
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
            state = tf.zeros([1, output_size])

            for time_step in range(paragraph_length):
                H_r = tf.reshape(knowledge_rep, [-1, 2 * output_size])
                F_s = tf.nn.tanh(tf.matmul(H_r, V) +
                                 tf.matmul(state, W_a) + b_a)
                probab_s = tf.reshape(tf.nn.softmax(
                    tf.matmul(F_s, v) + c), shape=[-1, paragraph_length])
                beta_s.append(probab_s)
                z = tf.matmul(probab_s, H_r)
                _, state = cell(z, state, scope="Boundary-LSTM_start")
                tf.get_variable_scope().reuse_variables()
        beta_s = tf.stack(beta_s)
        beta_s = tf.transpose(beta_s, perm=(1, 0, 2))

        # predict end index; beta_e is the probability distribution over the
        # paragraph words
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
            state = tf.zeros([1, output_size])
            for time_step in range(paragraph_length):
                H_r = tf.reshape(knowledge_rep, [-1, 2 * output_size])
                F_e = tf.nn.tanh(tf.matmul(H_r, V) +
                                 tf.matmul(state, W_a) + b_a)
                probab_e = tf.reshape(tf.nn.softmax(
                    tf.matmul(F_e, v) + c), shape=[-1, paragraph_length])
                beta_e.append(probab_e)

                z = tf.matmul(probab_e, H_r)
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
            # self.train_op_start = tf.train.AdamOptimizer()
            # self.train_op_end = tf.train.AdamOptimizer()

        # ==== set up training/updating procedure ====
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(
            self.config.learning_rate, step, 1, self.config.learning_rate_decay)

        optimizer = get_optimizer(self.config.optimizer)(rate)

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

        self.start_index_train_op = optimizer.apply_gradients(
            zip(c_s_grads, s_vars))
        self.end_index_train_op = optimizer.apply_gradients(
            zip(c_e_grads, e_vars))

    def setup_system(self):
        """
        Encode questions by using basic LSTM, also encodes paragraph using LSTM with attentions.
        After finishing encoding, it decodes the result using MatchLSTM, and setting preds, with matchLSTM results.
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
        """
        sets up losses based on decoder.decode inputs, refered as self.preds.
        preds[0] -> start index predictons, preds[1] -> end index predictions.
        returns losses for both of start_index and end_index.
        return start_index_loss, end_index_loss
        """
        preds = np.array(self.preds)

        with vs.variable_scope("start_index_loss"):
            loss_tensor = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=preds[0], labels=self.start_labels_placeholder), self.p_mask_placeholder)
            start_index_loss = tf.reduce_mean(loss_tensor, 0)
            tf.summary.scalar(
                'start_index_cross_entroy_loss', start_index_loss)

        with vs.variable_scope("end_index_loss"):
            loss_tensor = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=preds[1], labels=self.end_labels_placeholder), self.p_mask_placeholder)
            end_index_loss = tf.reduce_mean(loss_tensor, 0)
            tf.summary.scalar('end_index_cross_entroy_loss', end_index_loss)

        return start_index_loss, end_index_loss

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        Also adds some syntactic information such as postag, ner and exact match.
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
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = self.create_feed_dict(
            question_batch, context_batch, labels_batch, q_mask_batch, p_mask_batch)

        start_output_feed = [self.start_index_train_op, self.start_index_loss]
        _, start_index_loss = session.run(start_output_feed, input_feed)
        end_output_feed = [self.end_index_train_op, self.end_index_loss]
        _, end_index_loss = session.run(end_output_feed, input_feed)

        return start_index_loss, end_index_loss

    def test(self, session, valid_x, valid_y):
        input_feed = {}

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, train_x, mask):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

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
        NOTE: You do not have to do anything here.
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
            self.start_index_loss, self.end_index_loss = self.optimize(
                session, question_batch, context_batch, labels_batch, q_mask_batch, p_mask_batch)
            n_minibatches += 1

            losses.append([self.start_index_loss, self.start_index_loss])

        mean = np.mean(losses, axis=0)
        logging.info(
            "Logged mean epoch losses: train : %f dev : %f ", mean[0], mean[1])

        return losses

    def answer(self, session, test_x, mask):

        yp, yp2 = self.decode(session, test_x, mask)
        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)
        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.
        This method calls self.test() which explicitly calculates validation cost.
        How you implement this function is dependent on how you design
        your data iteration function
        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
            valid_cost = self.test(sess, valid_x, valid_y)

        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels
        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
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

        if log:
            logging.info(
                "F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop
        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.
        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one
        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.
        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.
        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
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
        print(dataset["Questions_masks"])
        for epoch in range(self.config.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.config.epochs)
            logging.info("Best score so far: " + str(best_score))
            loss = self.run_epoch(session, dataset)
            f1, em = self.evaluate_answer(
                session, dataset, sample=800, log=True)
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
