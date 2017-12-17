from synthesizer import Synthesizer
import csv
import re
import numpy as np



def eval_text(texts, checkpoint, out_path):
    synth = Synthesizer()
    synth.load(checkpoint)
    for i, text in enumerate(texts):
        path = '%s/%d.wav' % (out_path, i)
        print('%d test case, write to %s' % (i, path))
        with open(path, 'wb') as f:
          f.write(synth.synthesize(text))

def main():
    checkpoint = './checkpoint/model.ckpt-894000'
    out_path = './test'
    texts = []
    with open('./speech_texts') as f:
        for line in f:
            tokens = line.split(' ')
            #tokens = [re.sub(r'[^a-zA-Z0-9,\'.]+', ' ',token) for token in tokens]
            sentence = ' '.join(tokens)
            if len(tokens) >= 5 and len(tokens) <= 20:
                texts.append(sentence)
    test_texts = np.random.choice(texts, size=100, replace=False)
    with open('./test/test_texts', 'w+') as f:
        for text in test_texts:
            f.write(text)

    eval_text(test_texts, checkpoint, out_path)


if __name__ == '__main__':
    main()
