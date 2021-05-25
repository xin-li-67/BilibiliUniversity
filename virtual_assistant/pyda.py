import wikipedia
import wolframalpha
import pyttsx3
import PySimpleGUI as sg

client = wolframalpha.Client("lilpumpsaysnopeeking")
sg.theme('DarkPurple')
layout =[[sg.Text('Enter a command'), sg.InputText()],[sg.Button('Ok'), sg.Button('Cancel')]]
window = sg.Window('PyDa', layout)
engine = pyttsx3.init()

while True:
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
    
    try:
        wiki_res = wikipedia.summary(values[0], sentences=2)
        wolfram_res = next(client.query(values[0]).results).text
        engine.say(wolfram_res)
        sg.PopupNonBlocking("Wolfram Result: "+wolfram_res,"Wikipedia Result: "+wiki_res)
    except wikipedia.exceptions.DisambiguationError:
        wolfram_res = next(client.query(values[0]).results).text
        engine.say(wolfram_res)
        sg.PopupNonBlocking(wolfram_res)
    except wikipedia.exceptions.PageError:
        wolfram_res = next(client.query(values[0]).results).text
        engine.say(wolfram_res)
        sg.PopupNonBlocking(wolfram_res)
    except:
        wiki_res = wikipedia.summary(values[0], sentences=2)
        engine.say(wiki_res)
        sg.PopupNonBlocking(wiki_res)

    engine.runAndWait()
    print (values[0])

window.close()