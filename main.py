import nltk
import speech_recognition as sr

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('twitter_samples')

audio_file_path = "chile.wav"


def convert_audio_to_text(path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "google speech recognition couldn't understand the audio"
    except sr.RequestError as e:
        return f"the error is {e}"


text = convert_audio_to_text(audio_file_path)

analyzer = SentimentIntensityAnalyzer()
if analyzer.polarity_scores(text)['compound'] > 0:
    print("Positive text")
else:
    print("Negative text")
