from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello333():
    return "<h1>hello fuck world</h1>"

@app.route('/ping/', methods=['GET'])
def ping():
    return "<h1>ping<h1>"

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=False)    
    # app.run(host="192.168.0.178", port=8888, debug=False)    