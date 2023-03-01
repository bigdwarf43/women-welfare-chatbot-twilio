from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse
from helper.respond import response

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/bot', methods=['POST'])
def bot():
    incoming_msg = request.values.get('Body', '')
    print(incoming_msg)
    #print(incoming_msg)
    resp = MessagingResponse()
    msg = resp.message()
    text = response(incoming_msg)
    msg.body(text)
    return str(resp)

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)