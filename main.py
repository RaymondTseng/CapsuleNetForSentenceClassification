import tensorflow as tf
from tqdm import tqdm
from model import CapsuleNet
import data_helper
import utils

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9,  'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 30, 'epoch')
flags.DEFINE_integer('iter_ro-uting', 3, 'number of iterations in routing algorithm')
flags.DEFINE_integer('seq_length', 30, 'number of sequence length')
flags.DEFINE_integer('class_num', 2, 'number of classes')
flags.DEFINE_integer('filter_size', 1, 'number of filter size')
flags.DEFINE_integer('filter_num', 300, 'number of filters')
flags.DEFINE_integer('embedding_size', 300, 'number of embedding')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_boolean('if_train', True, 'train or evaluate')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')

flags.DEFINE_string('train_data_path', './data/mr.train.txt', 'the path of training dataset')
flags.DEFINE_string('test_data_path', './data/mr.test.txt', 'the path of test dataset')
flags.DEFINE_string('embedding_path', '/home/raymond/Downloads/data/glove.6B.300d.txt', 'the path of word embedding')
flags.DEFINE_string('log_dir', './log', 'the path of log dir')
FLAGS = tf.app.flags.FLAGS


def train(model, supervisor):
    test_sentences, test_labels = data_helper.load_data(FLAGS.test_data_path)
    test_sentences = utils.sentence2id(test_sentences, model.word2idx, FLAGS.seq_length)
    test_labels = utils.one_hot(test_labels, FLAGS.class_num)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        for epoch in range(FLAGS.epoch):
            print("Training for epoch %d/%d:" % (epoch, FLAGS.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            tqdm_iter = tqdm(range(model.train_num), total=model.train_num, leave=False)
            for step in tqdm_iter:
                _, loss, train_acc = sess.run([model.optimizer, model.total_loss, model.accuracy])
                tqdm_iter.set_description('--- loss: %.4f --- accuracy: %.4f ---' % (loss, train_acc))
            test_acc = sess.run(model.accuracy, {model.inputs: test_sentences, model.labels: test_labels})
            print('--- evaluate --- accuracy: %.4f ---' % test_acc)




def main(_):
    model = CapsuleNet(FLAGS)
    sv = tf.train.Supervisor(graph=model.graph)
    if FLAGS.if_train:
        train(model, sv)
if __name__ == '__main__':
    tf.app.run()
