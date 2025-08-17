from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)  # Qit the application if ne model does not load

conversation_histories = {}


def get_bot_response(input_text, history):
    """
    Generate a bot response based on user input and conversation history.

    Args:
        input_text (str): The user's current input (e.g., "What is AI?").
        history (list): A list of previous conversation turns
                        (e.g., ["Hi!", "Hello!"]).

    Returns:
        str: The bot's generated response.
    """
    try:
        # Limit the history to 10 exchanges
        # and create conversation history string
        history_string = "\n".join(
            [
                f"User: {h}" if i % 2 == 0 else f"Bot: {h}"
                for i, h in enumerate(history[-10:])
            ]
        )

        # Tokenize the input text and history
        full_text = f"{history_string}\nUser: {input_text}"
        inputs = tokenizer.encode_plus(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate the response from the model
        # max_length cause the model to crash at some point as history grows
        # max_time sets a timeout at 30s
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )

        # Return a decoded response
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    except Exception as e:
        print(f"Error in get_bot_response: {e}")  # Log l'erreur
        return "Désolé, une erreur est survenue."  # Response by default


@app.route('/', methods=['GET'])
def home():
    """
    Renders the home page of the application.

    Returns:
        flask.render_template: The rendered index.html template.
    """
    return render_template('index.html')


@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    """
    Handles POST requests to the chatbot endpoint.

    This function processes a JSON request containing a user prompt,
    generates a bot response, and updates the conversation history.

    Returns:
        flask.jsonify: A JSON response containing the bot's message.
                       Returns a 400 status code for invalid input or a
                       500 status code for server errors.
    """
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Invalid input"}), 400

        input_text = data['prompt']
        session_id = data.get('session_id', 'default')

        if session_id not in conversation_histories:
            conversation_histories[session_id] = []

        response = get_bot_response(
            input_text,
            conversation_histories[session_id])

        # Update the history
        conversation_histories[session_id].extend([input_text, response])

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
