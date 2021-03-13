import wolframalpha

input = raw_input("Q: ")
app_id = "WOLFRAMALPHA_ID"
client = wolframalpha.Client(app_id)

res = client.query(input)
ans = next(res.results).text

print(ans)