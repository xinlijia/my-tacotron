import os, re
import speech_recognition as sr
import difflib


def rating(true_result, predict_result):
    dif = 0
    for i,s in enumerate(difflib.ndiff(true_result, predict_result)):
        if s[0] != ' ':
            dif += 1
    print float(dif)/len(true_result)
    return float(dif)/len(true_result)

def main():
    texts = []
    with open('./test/test_texts', 'r') as f:
        for line in f:
            tokens = str(line)[:-1].split(' ')
            tokens = [token.lower() for token in tokens]
            tokens = [re.sub(r'[^a-zA-Z0-9]+', '',token) for token in tokens]
            texts.append(' '.join(tokens))
    results = []

    for i, text in enumerate(texts):
        audio_path = './test/%d.wav' % i
        AUDIO_FILE = (audio_path)

        # use the audio file as the audio source
        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)
        try:
            result = r.recognize_google(audio)
            # print result
            results.append(result)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
    sum_accuracy = 0
    for i in xrange(len(results)):
         sum_accuracy += rating(texts[i], results[i])
    print 1 - sum_accuracy/len(results)

if __name__ == '__main__':
    main()
