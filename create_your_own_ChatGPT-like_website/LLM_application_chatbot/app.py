from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_histories = {}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Invalid input"}), 400
        
        input_text = data['prompt']
        session_id = data.get('session_id', 'default')

        # Initialize the conversation history if necessary
        if session_id not in conversation_histories:
            conversation_histories[session_id]=[]
        
        # Limit the history to 10 exchanges
        history = conversation_histories[session_id][-10:]

        # Create conversation history string
        history_str = "\n".join(history)

        # Tokenize the input text and history
        #inputs = tokenizer.encode_plus(history_str, input_text, return_tensors="pt")
        full_text = f"{history_str}\nUser: {input_text}"
        inputs = tokenizer.encode_plus(
            full_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512  # Limit size to avoid overflow
        )

        # Generate the response from the model
        # max_length will cause the model to crash at some point as history grows
        # max_time sets a timeout at 30s
        outputs = model.generate(**inputs, max_length= 60, max_time=30)

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Update the history
        conversation_histories[session_id].extend([input_text, response])

        #return jsonify({'response': response})
        return response

    except Exception as e:
        return jsonify(f"error: {str(e)}"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)