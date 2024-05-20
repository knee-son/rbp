from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)

@app.route('/command', methods=['POST'])
def handle_command():
    request_data = request.get_json()
    command = request_data['command']
    comlist = command.split()
    print("terts nay ni chat nimo:", comlist)
    subprocess.run(comlist)
    return jsonify({'message': 'Command received'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
