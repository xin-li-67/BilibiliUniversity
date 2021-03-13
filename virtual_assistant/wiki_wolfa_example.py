import wikipedia
import wolframalpha

while True:
    input = raw_input("Q: ")

    try:
        #wolframalpha
        app_id = "WOLPHA_ID"
        client = wolframalpha.Client(app_id)
        res = client.query(input)
        ans = next(res.results).text
        print(ans)
    except:
        #wikipedia
        print(wikipedia.summary(input))