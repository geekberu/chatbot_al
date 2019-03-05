import tensorflow as tf
import numpy as np
import os
import time
import random
from disc.hier_rnn_model import Hier_rnn_model
import sys


def evaluate(session, model, config, evl_inputs, evl_labels, evl_masks):
    total_num = len(evl_inputs[0])

    fetches = [model.correct_num, model.prediction, model.logits, model.target]
    feed_dict = {}
    for i in range(config.max_len):
        feed_dict[model.input_data[i].name] = evl_inputs[i]
    feed_dict[model.target.name] = evl_labels
    feed_dict[model.mask_x.name] = evl_masks
    correct_num, prediction, logits, target = session.run(fetches, feed_dict)

    print("total_num: ", total_num)
    print("correct_num: ", correct_num)
    print("prediction: ", prediction)
    print("target: ", target)

    accuracy = float(correct_num) / total_num
    return accuracy


def hier_get_batch(config, max_set_len, query_set, answer_set, gen_set):
    batch_size = config.batch_size
    if batch_size % 2 == 1:
        return IOError("Error")
    train_query = []
    train_answer = []
    train_labels = []
    half_size = batch_size / 2
    is_random_choose = False
    if max_set_len > half_size:
        is_random_choose = True
    for i in range(int(half_size)):
        if is_random_choose:
            index = random.randint(0, max_set_len)
        else:
            index = i
        train_query.append(query_set[index])
        train_answer.append(answer_set[index])
        train_labels.append(1)
        train_query.append(query_set[index])
        train_answer.append(gen_set[index])
        train_labels.append(0)

    return train_query, train_answer, train_labels


def create_model(sess, config, vocab_size, name_scope, initializer=None):
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = Hier_rnn_model(config=config, vocab_size=vocab_size, name_scope=name_scope)
        disc_ckpt_dir = os.path.abspath(os.path.join(config.train_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(disc_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Hier Disc model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created Hier Disc model with fresh parameters.")
            disc_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            sess.run(tf.variables_initializer(disc_global_variables))
        return model


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


def hier_train(config_disc, config_evl, vocab_size, train_set):
    config_evl.keep_prob = 1.0

    print("Disc begin training...")

    with tf.Session() as session:

        query_set = train_set[0]
        answer_set = train_set[1]
        gen_set = train_set[2]

        train_bucket_sizes = [len(query_set[b]) for b in range(len(config_disc.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        total_qa_size = 0
        for i, set in enumerate(query_set):
            length = len(set)
            print("Discriminator train_set_{} len: {}".format(i, length))
            total_qa_size += length
        print("Discriminator train_set total size is {} QA".format(total_qa_size))

        model = create_model(session, config_disc, vocab_size, name_scope=config_disc.name_model)

        step_time, loss = 0.0, 0.0
        current_step = 0
        # previous_losses = []
        step_loss_summary = tf.Summary()
        disc_writer = tf.summary.FileWriter(config_disc.tensorboard_dir, session.graph)

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()

            b_query, b_answer, b_gen = query_set[bucket_id], answer_set[bucket_id], gen_set[bucket_id]

            train_query, train_answer, train_labels = hier_get_batch(config_disc, len(b_query)-1,
                                                                     b_query, b_answer, b_gen)

            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)

            feed_dict = {}
            for i in range(config_disc.buckets[bucket_id][0]):
                feed_dict[model.query[i].name] = train_query[i]
            for i in range(config_disc.buckets[bucket_id][1]):
                feed_dict[model.answer[i].name] = train_answer[i]
            feed_dict[model.target.name] = train_labels

            fetches = [model.b_train_op[bucket_id], model.b_logits[bucket_id], model.b_loss[bucket_id], model.target]
            train_op, logits, step_loss, target = session.run(fetches, feed_dict)

            step_time += (time.time() - start_time) / config_disc.steps_per_checkpoint
            loss += step_loss /config_disc.steps_per_checkpoint
            current_step += 1

            if current_step % config_disc.steps_per_checkpoint == 0:

                disc_loss_value = step_loss_summary.value.add()
                disc_loss_value.tag = config_disc.name_loss
                disc_loss_value.simple_value = float(loss)

                disc_writer.add_summary(step_loss_summary, int(session.run(model.global_step)))

                print("logits shape: ", np.shape(logits))

                # softmax operation
                logits = np.transpose(softmax(np.transpose(logits)))

                reward, gen_num = 0.0, 0
                for logit, label in zip(logits, train_labels):
                    if label == 0:
                        reward += logit[1]  # only for true probability
                        gen_num += 1
                # reward = reward / len(train_labels)
                reward = reward / gen_num
                print("reward: ", reward)

                print("current_step: %d, step_loss: %.4f" %(current_step, step_loss))

                if current_step % (config_disc.steps_per_checkpoint * 6) == 0:
                    print("current_step: %d, save_model" % (current_step))
                    disc_ckpt_dir = os.path.abspath(os.path.join(config_disc.train_dir, "checkpoints"))
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc_pretrain.model")
                    model.saver.save(session, disc_model_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

