import os
from hparams import hparams

import numpy as np
from util import audio


def build_from_path(in_path, out_path):
    '''
    for each wav files in the indexes(metadata.csv), call process_utterance()
    to generate spectrograms and save to files.
    return the metadata includes the np files paths and texts, which to be write to
    train.txt as an index
    '''
    futures = []
    index = 1
    with open(os.path.join(in_path, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_path, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            futures.append(process_utterance(out_path, index, wav_path, text))
            print ('%d/13100' % index)
            index += 1

    return [future for future in futures]


def process_utterance(out_path, index, wav_path, text):
    '''
    generate linear and mel scale spectrograms for each text, wav pairs
    and save the np array into disk

    return the file names of the np array files
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'ljspeech-spec-%05d.npy' % index
    mel_filename = 'ljspeech-mel-%05d.npy' % index

    # .T: transpose of narray
    # allow_pickle: for security and portability not allow
    np.save(os.path.join(out_path, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_path, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text)

def write_metadata(metadata, out_path):
    with open(os.path.join(out_path, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    hours = frames * hparams.frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))



def preprocess(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    metadata = build_from_path(in_path, out_path)
    write_metadata(metadata, out_path)




def main():
    in_path = '../my-tacotron/LJSpeech-1.0'
    out_path = './training'
    preprocess(in_path, out_path)

if __name__ == '__main__':
    main()
