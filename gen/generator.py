from __future__ import division
from __future__ import print_function

import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import utils.conf as conf
import gen.gen_model as seq2seq_model
import nltk  # For tokenize

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.

SENTENCES_PREFIX = ['Q: ', 'A: ']


# create the train_set, format: [bucket_id][source_ids, target_ids]
def create_train_set(config, text_data):
    data_set = [[] for _ in config.buckets]
    samples = text_data.trainingSamples
    for sample in samples:
        source = sample[0]
        target = sample[1]
        # source_ids = [int(x) for x in source.split()]
        # target_ids = [int(x) for x in target.split()]
        for bucket_id, (source_size, target_size) in enumerate(
                config.buckets):  # [bucket_id, (source_size, target_size)]
            if len(source) < source_size and len(target) < (target_size - 2):
                data_set[bucket_id].append([source, target])
                break
    return data_set


def create_disc_train_set(config, text_data, bucket_id=-1, train_set=None, batch_num=1, sess=None, gen_model=None):
    if train_set is None:
        train_set = create_train_set(config, text_data)
    random_bucket_id = False
    if bucket_id is -1:
        train_bucket_sizes = [len(train_set[b]) for b in range(len(config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]
        random_bucket_id = True

    query_set = [[] for _ in config.buckets]
    answer_set = [[] for _ in config.buckets]
    gen_set = [[] for _ in config.buckets]

    model = gen_model
    is_close_sess = False
    if sess is None:
        sess = tf.Session()
        model = create_model(sess, config, text_data.getVocabularySize(), forward_only=True,
                             name_scope=config.name_model)
        is_close_sess = True

    num_step = 0
    print("total generating steps: ", batch_num)
    while num_step < batch_num:
        print("generating num_step: ", num_step)
        if random_bucket_id:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

        encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = \
            get_batch(config, train_set, bucket_id, config.batch_size, text_data)

        _, _, out_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                      forward_only=True)
        seq_tokens = []
        for seq in out_logits:
            row_token = []
            for t in seq:
                row_token.append(int(np.argmax(t, axis=0)))
            seq_tokens.append(row_token)

        seq_tokens_t = []
        for col in range(len(seq_tokens[0])):
            seq_tokens_t.append([seq_tokens[row][col] for row in range(len(seq_tokens))])

        for i in range(config.batch_size):
            query_set[bucket_id].append(batch_source_encoder[i])
            answer_set[bucket_id].append(batch_source_decoder[i])
            gen_set[bucket_id].append(seq_tokens_t[i])

        num_step += 1

    train_set = [query_set, answer_set, gen_set]
    if is_close_sess:
        sess.close()
    return train_set


def create_model(session, gen_config, vocab_size, forward_only, name_scope, initializer=None):
    """Create translation model and initialize or load parameters in session."""
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = seq2seq_model.Seq2SeqModel(gen_config, vocab_size=vocab_size, name_scope=name_scope,
                                           forward_only=forward_only)
        gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(gen_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Gen model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created Gen model with fresh parameters.")
            gen_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            session.run(tf.variables_initializer(gen_global_variables))
        return model


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


def get_batch(config, train_set, bucket_id, batch_size, text_data):
    # Q_size, A_size = config.buckets[bucket_id]
    encoder_size, decoder_size = config.buckets[bucket_id]
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_source_encoder, batch_source_decoder, samples = [], [], []
    # print("bucket_id: %s" %bucket_id)
    for batch_i in range(batch_size):
        encoder_input, decoder_input = random.choice(train_set[bucket_id])
        sample = [encoder_input, decoder_input]
        samples.append(sample)
        encoder_input = text_data.add_pad(encoder_input, config.buckets[bucket_id][0])
        batch_source_encoder.append(encoder_input)
        decoder_input = text_data.add_pad(decoder_input, config.buckets[bucket_id][1])
        batch_source_decoder.append(decoder_input)

    # Now we create batch-major vectors from the disc_data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # batch = text_data.get_batch(samples, Q_size, A_size)
    batch = text_data.get_batch(samples, encoder_size, decoder_size)
    batch_encoder_inputs = batch.encoderSeqs
    batch_decoder_inputs = batch.decoderSeqs
    batch_weights = batch.weights

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder


def train(gen_config, text_data):
    # vocab, rev_vocab, train_set = prepare_data(gen_config)
    train_set = create_train_set(gen_config, text_data)

    total_qa_size = 0
    for i, set in enumerate(train_set):
        length = len(set)
        print("Generator train_set_{} len: {}".format(i, length))
        total_qa_size += length
    print("Generator train_set total size is {} QA".format(total_qa_size))

    with tf.Session() as sess:
    #with tf.device("/gpu:1"):
        # Create model.
        print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.emb_dim))
        vocab_size = text_data.getVocabularySize()
        model = create_model(sess, gen_config, vocab_size, forward_only=False,
                             name_scope=gen_config.name_model)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        # previous_losses = []

        gen_loss_summary = tf.Summary()
        gen_writer = tf.summary.FileWriter(gen_config.tensorboard_dir, sess.graph)

        while True:
            # Choose a bucket according to disc_data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, _, _ = get_batch(gen_config, train_set, bucket_id,
                                                                             gen_config.batch_size, text_data)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            loss += step_loss / gen_config.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gen_config.steps_per_checkpoint == 0:

                bucket_value = gen_loss_summary.value.add()
                bucket_value.tag = gen_config.name_loss
                bucket_value.simple_value = float(loss)
                gen_writer.add_summary(gen_loss_summary, int(model.global_step.eval()))

                # Print statistics for the previous epoch.
                # perplexity = math.exp(loss) if loss < 300 else float('inf')
                # print("global step %d learning rate %.4f step-time %.2f perplexity "
                #        "%.2f" % (model.global_step.eval(), gen_config.learning_rate,
                #                  step_time, perplexity))
                print("global step %d learning rate %.4f step-time %.2f loss "
                       "%.2f" % (model.global_step.eval(), gen_config.learning_rate,
                                 step_time, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                #     sess.run(model.learning_rate_decay_op)
                # previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                if current_step % (gen_config.steps_per_checkpoint * 6) == 0:
                    print("current_step: %d, save model" %(current_step))
                    gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
                    if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                    checkpoint_path = os.path.join(gen_ckpt_dir, "gen_pretrain.model")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                # for bucket_id in range(len(gen_config.buckets)):
                #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                #       dev_set, bucket_id)
                #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                #                                target_weights, bucket_id, True)
                #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def inference_interactive(text_data):
    config = conf.gen_config
    with tf.Session() as sess:
        model = create_model(sess, config, text_data.getVocabularySize(), forward_only=True, name_scope=config.name_model)
        model.batch_size = 1
        # print('Testing: Launch interactive mode:')
        print('**************************************************************************************')
        print('*  Welcome to the interactive mode, here you can ask Chatbot the sentence you want.  *\n'
              '*  Don\'t have high expectation.                                                      *\n'
              '*  Type \'exit\' or just press ENTER to quit the program. Have fun.                    *')
        print('**************************************************************************************')
        while True:
            question = input(SENTENCES_PREFIX[0])
            if question == '' or question == 'exit':
                break
            # First step: Divide the sentence in token
            tokens = nltk.word_tokenize(question)
            # Second step: Convert the token in word ids
            word_ids = []
            bucket_id = len(config.buckets) - 1
            for token in tokens:
                word_ids.append(text_data.getWordId(token, create=False))  # Create the vocabulary and the training sentences
            for i, bucket in enumerate(config.buckets):
                if bucket[0] >= len(word_ids):
                    bucket_id = i
                    break
            else:
                print('Warning: sentence too long, sorry. Maybe try a shorter sentence.')

            samples = []
            sample = [word_ids, []]
            samples.append(sample)
            # Q_size, A_size = config.buckets[bucket_id]
            # batch = text_data.get_batch(samples, Q_size, A_size)
            encoder_size, decoder_size = config.buckets[bucket_id]
            batch = text_data.get_batch(samples, encoder_size, decoder_size)
            encoder_inputs = batch.encoderSeqs
            decoder_inputs = batch.decoderSeqs
            weights = batch.weights
            start = time.time()
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, weights, bucket_id, True)
            end = time.time()
            process_time = end - start

            print("output_logits shape: ", np.shape(output_logits))
            print("inference time: ", process_time)

            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            outputs = text_data.sequence2str(outputs, clean=True)
            print('{}{}'.format(SENTENCES_PREFIX[1], outputs))

