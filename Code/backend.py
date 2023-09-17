from flask import Flask, request, jsonify, render_template
from subprocess import run, PIPE

app = Flask(__name__, template_folder='template')
output_list = []

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        message = request.form.get('message')
        run_bot(message)
        return jsonify(response=output_list)
    elif request.method == 'GET':
        return render_template('index.html')

def run_bot(msg):
    process = run(['python3', 'finvisebot_team10.py'], input=msg, stdout=PIPE, stderr=PIPE, text=True)
    output = process.stdout.strip()
    output_list.append(output)
    print(output)
    return output
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    bot_response = run_bot(user_input)  # Llama a la función para procesar el chatbot y obtener la respuesta
    return {'response': bot_response}


if __name__ == '__main__':
    app.run()