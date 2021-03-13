# must install pyaudio correctly first
# then install SpeechRecognition

import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    audio = recognizer.listen(source)

output = recognizer.recognize_google(audio)

print(output)