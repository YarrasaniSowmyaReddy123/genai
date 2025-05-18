pip install SpeechRecognition pyttsx3 pyaudio
import speech_recognition as sr
import pyttsx3

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
tts = pyttsx3.init()

def speak(text):
    print("Bot:", text)
    tts.say(text)
    tts.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening... (Speak now)")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print("You:", text)
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that.")
        except sr.RequestError:
            speak("Speech service is unavailable.")
        return None

def chatbot_response(user_input):
    # Simple rule-based response (replace with your own logic or API)
    if "hello" in user_input.lower():
        return "Hello! How can I help you?"
    elif "how are you" in user_input.lower():
        return "I'm just a program, but I'm doing well!"
    else:
        return "Sorry, I didn't understand that."

# Main loop
while True:
    user_input = listen()
    if user_input:
        response = chatbot_response(user_input)
        speak(response)
