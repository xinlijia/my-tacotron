import time
import math
import os
from datetime import datetime

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from datafeeder import DataFeeder
from models import create_model
from util import audio, infolog, plot
from text import sequence_to_text

log = infolog.log


def add_stats(model):
    with tf.variable_scope('stats') as scope:
        tf.summary.histogram('linear_outputs', model.linear_outputs)
        tf.summary.histogram('linear_targets', model.linear_targets)
        tf.summary.histogram('mel_outputs', model.mel_outputs)
        tf.summary.histogram('mel_targets', model.mel_targets)
        tf.summary.scalar('loss_mel', model.mel_loss)
        tf.summary.scalar('loss_linear', model.linear_loss)
        tf.summary.scalar('learning_rate', model.learning_rate)
        tf.summary.scalar('loss', model.loss)
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
        return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')

def train(log_dir, input_path, checkpoint_path, is_restore):
    # Log the info
    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_path)
    log(hparams_debug_string())

    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        feeder = DataFeeder(coord, input_path, hparams)

    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('model') as scope:
        model = create_model('tacotron', hparams)
        model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.linear_targets)
        model.add_loss()
        model.add_optimizer(global_step)
        stats = add_stats(model)

    # Bookkeeping:
    step = 0
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
    # Train!
    with tf.Session() as sess:
        try:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            if is_restore:
                # Restore from a checkpoint if the user requested it.
                restore_path = '%s' % (checkpoint_path)
                saver.restore(sess, restore_path)
                log('Resuming from checkpoint')
            else:
                log('Starting new training')

            feeder.start_in_session(sess)

            while not coord.should_stop():
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.optimize])
                time_interval = time.time() - start_time

                message = 'Step %d, %.03f sec, loss=%.05f' % (step, loss, time_interval)
                log(message)

                if loss > 100 or math.isnan(loss):
                    log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
                    raise Exception('Loss Exploded')

                if step % hparams.summary_interval == 0:
                    log('Writing summary at step: %d' % step)
                    summary_writer.add_summary(sess.run(stats), step)

                if step % hparams.checkpoint_interval == 0:
                    log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                    saver.save(sess, checkpoint_path, global_step=step)
                    log('Saving audio and alignment...')
                    input_seq, spectrogram, alignment = sess.run([
                        model.inputs[0], model.linear_outputs[0], model.alignments[0]])
                    waveform = audio.inv_spectrogram(spectrogram.T)
                    audio.save_wav(waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
                    plot.plot_alignment(alignment, os.path.join(log_dir, 'step-%d-align.png' % step),
                        info='%s, %s, step=%d, loss=%.5f' % ('tacotron', time_string(), step, loss))
                    log('Input: %s' % sequence_to_text(input_seq))

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            coord.request_stop(e)

def main():
    log_dir = './logs'
    checkpoint_path = './checkpoint/model.ckpt'
    input_path = 'training/train.txt'
    is_restore = True
    train(log_dir, input_path, checkpoint_path, is_restore)



if __name__ == '__main__':
    main()
