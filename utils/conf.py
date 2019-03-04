__author__ = 'liuyuemaicha'
import os


class disc_config(object):
    # batch_size = 256
    batch_size = 16
    lr = 0.001
    lr_decay = 0.9
    embed_dim = 512
    steps_per_checkpoint = 100
    #hidden_neural_size = 128
    num_layers = 2
    train_dir = './disc_data/'
    name_model = "disc_model"
    tensorboard_dir = "./tensorboard/disc_log/"
    name_loss = "disc_loss"
    max_len = 50
    piece_size = batch_size * steps_per_checkpoint
    piece_dir = "./disc_data/batch_piece/"
    #query_len = 0
    valid_num = 100
    init_scale = 0.1
    num_class = 2
    keep_prob = 0.5
    #num_epoch = 60
    #max_decay_epoch = 30
    max_grad_norm = 5
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    epoch_num = 100


class gen_config(object):
    # batch_size = 128
    batch_size = 8
    beam_size = 7
    learning_rate = 0.001
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    disc_data_batch_num = 100
    emb_dim = 512
    num_layers = 2
    train_dir = "./gen_data/"
    name_model = "gen_model"
    tensorboard_dir = "./tensorboard/gen_log/"
    name_loss = "gen_loss"
    teacher_loss = "teacher_loss"
    reward_name = "reward"
    max_train_data_size = 0
    steps_per_checkpoint = 100
    # bucket->(source_size, target_size), source is the query, target is the answer
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]



