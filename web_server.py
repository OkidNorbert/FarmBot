
from flask import Flask, render_template, request, jsonify
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <html>
    <head><title>🍅 Tomato Sorter</title></head>
    <body>
        <h1>🍅 AI Tomato Sorter</h1>
        <p>Web interface is running!</p>
        <p>Model: tomato_classifier.pth</p>
        <p>Status: Ready for inference</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
