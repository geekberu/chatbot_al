import os

import tensorflow as tf
import numpy as np
import sys
import time
import gen.generator as gens
import disc.hier_disc as h_disc
import random
import utils.conf as conf
import argparse  # Command line parsing
from corpus.textdata import TextData

import connection

gen_config = conf.gen_config
disc_config = conf.disc_config
evl_config = conf.disc_config

G_STEPS = 1
D_STEPS = 5

# text_data = None


# pre train discriminator
def disc_pre_train(text_data):
    train_set = gens.create_disc_train_set(gen_config, text_data, -1, None, gen_config.disc_data_batch_num)
    h_disc.hier_train(disc_config, evl_config, text_data.getVocabularySize(), train_set)


# pre train generator
def gen_pre_train(text_data):
    gens.train(gen_config, text_data)


# test gen model
def gen_test_interactive(text_data):
    gens.inference_interactive(text_data)


def get_negative_decoder_inputs(sess, gen_model, encoder_inputs, decoder_inputs,
                                target_weights, bucket_id, mc_search=False):
    _, _, out_logits = gen_model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                      forward_only=True, mc_search=mc_search)
    seq_tokens = []
    for seq in out_logits:
        row_token = []
        for t in seq:
            row_token.append(int(np.argmax(t, axis=0)))
        seq_tokens.append(row_token)

    seq_tokens_t = []
    for col in range(len(seq_tokens[0])):
        seq_tokens_t.append([seq_tokens[row][col] for row in range(len(seq_tokens))])

    return seq_tokens_t


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


# discriminator api
def disc_step(sess, bucket_id, disc_model, train_query, train_answer, train_labels, forward_only=False):
    feed_dict = {}

    for i in range(len(train_query)):

        feed_dict[disc_model.query[i].name] = train_query[i]

    for i in range(len(train_answer)):
        feed_dict[disc_model.answer[i].name] = train_answer[i]

    feed_dict[disc_model.target.name] = train_labels

    loss = 0.0
    if forward_only:
        fetches = [disc_model.b_logits[bucket_id]]
        logits = sess.run(fetches, feed_dict)
        logits = logits[0]
    else:
        fetches = [disc_model.b_train_op[bucket_id], disc_model.b_loss[bucket_id], disc_model.b_logits[bucket_id]]
        train_op, loss, logits = sess.run(fetches, feed_dict)

    # softmax operation
    logits = np.transpose(softmax(np.transpose(logits)))

    reward, gen_num = 0.0, 0
    for logit, label in zip(logits, train_labels):
        # if label == 0 that means the answer is machine generated,
        # logit[0] means the 0's probability, logit[1] means the 1's probability
        # so when label == 0, we get logit[1] means the disc discriminate the machine generated answer is human's probability
        # and use it as reward for generator training
        if int(label) == 0:
            reward += logit[1]
            gen_num += 1
    reward = reward / gen_num

    return reward, loss


# Adversarial Learning for Neural Dialogue Generation
def al_train(text_data):
    with tf.Session() as sess:
        train_set = gens.create_train_set(gen_config, text_data)

        total_qa_size = 0
        for i, set in enumerate(train_set):
            length = len(set)
            print("Generator train_set_{} len: {}".format(i, length))
            total_qa_size += length
        print("Generator train_set total size is {} QA".format(total_qa_size))

        train_bucket_sizes = [len(train_set[b]) for b in range(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]
        vocab_size = text_data.getVocabularySize()
        disc_model = h_disc.create_model(sess, disc_config, vocab_size, disc_config.name_model)
        gen_model = gens.create_model(sess, gen_config, vocab_size, forward_only=False,
                                      name_scope=gen_config.name_model)

        current_step = 0
        step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
        gen_loss_summary = tf.Summary()
        disc_loss_summary = tf.Summary()

        gen_writer = tf.summary.FileWriter(gen_config.tensorboard_dir, sess.graph)
        disc_writer = tf.summary.FileWriter(disc_config.tensorboard_dir, sess.graph)

        while True:
            current_step += 1
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
            print("==================Update Discriminator: %d==================" % current_step)
            for i in range(D_STEPS):
                print("=============It's the %d time update Discriminator in current step=============" % (i+1))

                # 1. Sample (X,Y) from real data and sample ^Y from G(*|X)
                query_set, answer_set, gen_set = gens.create_disc_train_set(gen_config, text_data, bucket_id,
                                                                            train_set, 1, sess, gen_model)

                b_query, b_answer, b_gen = query_set[bucket_id], answer_set[bucket_id], gen_set[bucket_id]

                train_query, train_answer, train_labels = h_disc.hier_get_batch(disc_config, len(b_query) - 1,
                                                                                b_query, b_answer, b_gen)
                train_query = np.transpose(train_query)
                train_answer = np.transpose(train_answer)

                # 2. Update D using (X,Y) as positive examples and(X,^Y) as negative examples
                _, disc_step_loss = disc_step(sess, bucket_id, disc_model, train_query, train_answer,
                                              train_labels, forward_only=False)
                disc_loss += disc_step_loss / (D_STEPS * disc_config.steps_per_checkpoint)
                if i == D_STEPS - 1:
                    print("disc_step_loss: ", disc_step_loss)

            print("==================Update Generator: %d==================" % current_step)
            for j in range(G_STEPS):
                print("=============It's the %d time update Generator in current step=============" % (j+1))
                # 1. Sample (X,Y) from real data
                encoder_inputs, decoder_inputs, target_weights,\
                    source_inputs, source_outputs = gens.get_batch(gen_config, train_set, bucket_id,
                                                                   gen_config.batch_size, text_data)

                # 2. Sample ^Y from G(*|X) for generator update
                decoder_inputs_negative = get_negative_decoder_inputs(sess, gen_model, encoder_inputs,
                                                                      decoder_inputs, target_weights, bucket_id)
                decoder_inputs_negative = np.transpose(decoder_inputs_negative)

                # 3. Sample ^Y from G(*|X) with Monte Carlo search
                train_query, train_answer, train_labels = [], [], []
                for query, answer in zip(source_inputs, source_outputs):
                    train_query.append(query)
                    train_answer.append(answer)
                    train_labels.append(1)
                for _ in range(gen_config.beam_size):
                    gen_set = get_negative_decoder_inputs(sess, gen_model, encoder_inputs, decoder_inputs,
                                                          target_weights, bucket_id, mc_search=True)
                    for i, output in enumerate(gen_set):
                        train_query.append(train_query[i])
                        train_answer.append(output)
                        train_labels.append(0)

                train_query = np.transpose(train_query)
                train_answer = np.transpose(train_answer)

                # 4. Compute Reward r for (X,^Y) using D.---based on Monte Carlo search
                reward, _ = disc_step(sess, bucket_id, disc_model, train_query, train_answer,
                                      train_labels, forward_only=True)
                batch_reward += reward / gen_config.steps_per_checkpoint
                print("step_reward: ", reward)

                # 5. Update G on (X,^Y) using reward r
                gan_adjusted_loss, gen_step_loss, _ = gen_model.step(sess, encoder_inputs, decoder_inputs_negative,
                                                                     target_weights, bucket_id, forward_only=False,
                                                                     reward=reward, up_reward=True, debug=True)
                gen_loss += gen_step_loss / gen_config.steps_per_checkpoint

                print("gen_step_loss: ", gen_step_loss)
                print("gen_step_adjusted_loss: ", gan_adjusted_loss)

                # 6. Teacher-Forcing: Update G on (X,Y)
                t_adjusted_loss, t_step_loss, a = gen_model.step(sess, encoder_inputs, decoder_inputs,
                                                                 target_weights, bucket_id, forward_only=False)
                t_loss += t_step_loss / (G_STEPS * gen_config.steps_per_checkpoint)

                print("t_step_loss: ", t_step_loss)
                print("t_adjusted_loss", t_adjusted_loss)

            if current_step % gen_config.steps_per_checkpoint == 0:

                step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint

                print("current_steps: %d, step time: %.4f, disc_loss: %.3f, gen_loss: %.3f, t_loss: %.3f, reward: %.3f "
                      % (current_step, step_time, disc_loss, gen_loss, t_loss, batch_reward))

                disc_loss_value = disc_loss_summary.value.add()
                disc_loss_value.tag = disc_config.name_loss
                disc_loss_value.simple_value = float(disc_loss)
                disc_writer.add_summary(disc_loss_summary, int(sess.run(disc_model.global_step)))

                gen_global_steps = sess.run(gen_model.global_step)
                gen_loss_value = gen_loss_summary.value.add()
                gen_loss_value.tag = gen_config.name_loss
                gen_loss_value.simple_value = float(gen_loss)
                t_loss_value = gen_loss_summary.value.add()
                t_loss_value.tag = gen_config.teacher_loss
                t_loss_value.simple_value = float(t_loss)
                batch_reward_value = gen_loss_summary.value.add()
                batch_reward_value.tag = gen_config.reward_name
                batch_reward_value.simple_value = float(batch_reward)
                gen_writer.add_summary(gen_loss_summary, int(gen_global_steps))

                if current_step % (gen_config.steps_per_checkpoint * 4) == 0:
                    print("current_steps: %d, save disc model" % current_step)
                    disc_ckpt_dir = os.path.abspath(os.path.join(disc_config.train_dir, "checkpoints"))
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                    disc_model.saver.save(sess, disc_model_path, global_step=disc_model.global_step)

                    print("current_steps: %d, save gen model" % current_step)
                    gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
                    if not os.path.exists(gen_ckpt_dir):
                        os.makedirs(gen_ckpt_dir)
                    gen_model_path = os.path.join(gen_ckpt_dir, "gen.model")
                    gen_model.saver.save(sess, gen_model_path, global_step=gen_model.global_step)

                step_time, disc_loss, gen_loss, t_loss, batch_reward = 0.0, 0.0, 0.0, 0.0, 0.0
                sys.stdout.flush()


def parse_args():
    """
    Parse the arguments from the given command line
    Args:
        args (list<str>): List of arguments to parse. If None, the default sys.argv will be parsed
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=0, help='Test mode')
    parser.add_argument('--maxLength', type=int, default=40,
                        help='Maximum length of the sentence (for input and output), define number of maximum step of the RNN')
    parser.add_argument('--filterVocab', type=int, default=1,
                        help='Remove rarely used words (by default words used only once). 0 to keep all words.')
    parser.add_argument('--vocabularySize', type=int, default=40000,
                        help='Limit the number of words in the vocabulary (0 for unlimited)')
    parser.add_argument('--corpus', choices=TextData.corpusChoices(), default=TextData.corpusChoices()[0],
                        help='Corpus on which extract the dataset.')
    parser.add_argument('--rootDir', type=str, default='corpus', help='Folder where to look for the models and data')
    parser.add_argument('--datasetTag', type=str, default='',
                        help='Add a tag to the dataset (file where to load the vocabulary and the precomputed samples, not the original corpus). Useful to manage multiple versions. Also used to define the file used for the lightweight format.')  # The samples are computed from the corpus if it does not exist already. There are saved in \'data/samples/\'
    parser.add_argument('--skipLines', action='store_true', default=True,
                        help='Generate training samples by only using even conversation lines as questions (and odd lines as answer). Useful to train the network on a particular person.')
    args = parser.parse_args()
    return args


def main():
    # global text_data
    args = parse_args()
    text_data = TextData(args)
    try:
        if args.test:
            gen_test_interactive(text_data)
        else:
            # Step 1: Pre train the Generator and get the GEN_0 model
            gen_pre_train(text_data)

            # Step 2: GEN model test
            # gen_test_interactive(text_data)

            # Step 3: Pre train the Discriminator and get the DISC_0 model
            # disc_pre_train(text_data)

            # Step 4: Train the GEN model and DISC model using AL/RL
            # al_train(text_data)

            # Step 5: GEN model test
            # gen_test_interactive(text_data)

            # integration test
            # connection.start_server(text_data, True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
