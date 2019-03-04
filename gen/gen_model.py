from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
import gen.seq2seq as rl_seq2seq


class Seq2SeqModel(object):

    def __init__(self, config, vocab_size, name_scope, forward_only=False, num_samples=512, dtype=tf.float32):

        # self.scope_name = scope_name
        # with tf.variable_scope(self.scope_name):
        source_vocab_size = vocab_size
        target_vocab_size = vocab_size
        emb_dim = config.emb_dim

        self.buckets = config.buckets
        # self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False, dtype=dtype)
        self.learning_rate = config.learning_rate
        # self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers
        self.max_gradient_norm = config.max_gradient_norm
        self.mc_search = tf.placeholder(tf.bool, name="mc_search")
        self.forward_only = tf.placeholder(tf.bool, name="forward_only")
        self.up_reward = tf.placeholder(tf.bool, name="up_reward")
        self.reward_bias = tf.get_variable("reward_bias", [1], dtype=tf.float32)
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < target_vocab_size:
            w_t = tf.get_variable("proj_w", [target_vocab_size, emb_dim], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                    # tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                    #                            num_samples, target_vocab_size), dtype)
                    tf.nn.sampled_softmax_loss(local_w_t, local_b, labels, local_inputs,
                                               num_samples, target_vocab_size), dtype)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(emb_dim)
        cell = single_cell
        if self.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return rl_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols= source_vocab_size,
                num_decoder_symbols= target_vocab_size,
                embedding_size= emb_dim,
                output_projection=output_projection,
                feed_previous=do_decode,
                mc_search=self.mc_search,
                dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(self.buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        # for i in range(self.buckets[-1][1] + 2 + 1):
        for i in range(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
        self.reward = [tf.placeholder(tf.float32, name="reward_%i" % i) for i in range(len(self.buckets))]

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        self.outputs, self.losses, self.encoder_state = rl_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets, self.target_weights,
            self.buckets, source_vocab_size, self.batch_size,
            lambda x, y: seq2seq_f(x, y, tf.where(self.forward_only, True, False)),
            output_projection=output_projection, softmax_loss_function=softmax_loss_function)

        for b in range(len(self.buckets)):
            self.outputs[b] = [
                tf.cond(
                    self.forward_only,
                    lambda: tf.matmul(output, output_projection[0]) + output_projection[1],
                    lambda: output
                )
                for output in self.outputs[b]
            ]
        
        if not forward_only:
            with tf.name_scope("gradient_descent"):
                self.gradient_norms = []
                self.updates = []
                self.aj_losses = []
                self.gen_params = [p for p in tf.trainable_variables() if name_scope in p.name]
                #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                opt = tf.train.AdamOptimizer(self.learning_rate)
                for b in range(len(self.buckets)):
                    R = tf.subtract(self.reward[b], self.reward_bias)
                    # self.reward[b] = self.reward[b] - reward_bias
                    adjusted_loss = tf.cond(self.up_reward,
                                              lambda:tf.multiply(self.losses[b], self.reward[b]),
                                              lambda: self.losses[b])

                    # adjusted_loss =  tf.cond(self.up_reward,
                    #                           lambda: tf.mul(self.losses[b], R),
                    #                           lambda: self.losses[b])
                    self.aj_losses.append(adjusted_loss)
                    gradients = tf.gradients(adjusted_loss, self.gen_params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, self.gen_params), global_step=self.global_step))

        self.gen_variables = [k for k in tf.global_variables() if name_scope in k.name]
        self.saver = tf.train.Saver(self.gen_variables)

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only=True, reward=1, mc_search=False, up_reward=False, debug=True):
        # Check if the sizes match.
        # Q_size, A_size = self.buckets[bucket_id]
        # encoder_size = Q_size
        # decoder_size = A_size + 2
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.

        input_feed = {
            self.forward_only.name: forward_only,
            self.up_reward.name:  up_reward,
            self.mc_search.name: mc_search
        }
        for l in range(len(self.buckets)):
            input_feed[self.reward[l].name] = reward
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only: # normal training
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                       self.aj_losses[bucket_id],  # Gradient norm.
                       self.losses[bucket_id]]  # Loss for this batch.
        else: # testing or reinforcement learning
            output_feed = [self.encoder_state[bucket_id], self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[0]  # Gradient norm, loss, no outputs.
        else:
            return outputs[0], outputs[1], outputs[2:]  # encoder_state, loss, outputs.
