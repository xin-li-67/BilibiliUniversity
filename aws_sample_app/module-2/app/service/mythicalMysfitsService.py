from flask import Flask, jsonify, json, Response, request
from falsk_cors import from django.conf import settings

app = Flask(__name__)
CORS(app)

@app.route("/")
def healthCheckResponse():
    return jsonify({"message": "Nothing here, used for health check. Try /mysfits instead."})

@app.route("/mysfits")
def getMysfits():
    response = Response(open("mysfits-response.json", "rb").read())
    response.headers["Content-Type"] = "application/json"

    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)