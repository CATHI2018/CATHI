'''
YUEXIAOLI---CATHI
'''

import os
import tensorflow as tf
from configs.config import TEST_DATASET_PATH, FLAGS
from lib import data_utils
from lib.seq2seq_model_utils import create_model, get_predicted_sentence


def Trajectory_Prediction():
    def _get_test_dataset():
        with open(TEST_DATASET_PATH) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences

    results_filename = '_'.join(['results_right', str(FLAGS.num_layers), str(FLAGS.size), str(FLAGS.vocab_size)])
    results_path = os.path.join(FLAGS.results_dir, results_filename)

    with tf.Session() as sess, open(results_path, 'w') as results_fh:
        model = create_model(sess, forward_only=True)
        model.batch_size = 1  # We decode one sentence at a time.


        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        test_dataset = _get_test_dataset()

        i = 0
        j = 0
        allright = 0
        for sentence in test_dataset:
            if i % 2 == 0:
                predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, sess)
                predicted_sentence =  sentence+ ' ' + predicted_sentence
                print(sentence, ' -> ', predicted_sentence)
                end = sentence
            if i % 2 == 1:

                sentence =  end+ ' ' +sentence
                if sentence == predicted_sentence:
                    allright += 1
                    print('^ is allright' + '\n')
                else:
                    print('Error~right is %s' % sentence)
                results_fh.write(sentence + '\n' + predicted_sentence + '\n')
            i = i + 1
        print 'traj=', i / 2, ',allright=', allright, ',accuracy=', allright * 1.0 / (i * 1.0 / 2)
        results_fh.write('traj=%d,allright=%d,accuracy=%f' % (i / 2, allright, allright * 1.0 / (i * 1.0 / 2)))