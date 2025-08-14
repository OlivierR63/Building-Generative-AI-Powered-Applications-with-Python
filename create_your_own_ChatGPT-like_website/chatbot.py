from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []
print("\nBot: Hi! Ask me a question or type 'quit' to end the chat")

while True:
    try:
        # Create conversation history string
        history_string = "\n".join([f"User: {h}" if i % 2 == 0 else f"Bot: {h}" for i, h in enumerate(conversation_history[-4:])])

        # Get the input data from the user
        input_text = input("\nQuestion > ").strip()

        if not input_text:
            continue
        
        if input_text.lower() == "quit":
            print("\nBot: Au revoir !")
            break

        # Concatenate the history and the new question into only one string
        full_text = f"{history_string}\nUser: {input_text}"
        inputs = tokenizer.encode_plus(
            full_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512  # Limit size to avoid overflow
        )

        # Generate the response from the model
        outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(f"\nBot: {response}")

        # Add interaction to conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)

    except KeyboardInterrupt:
        print("\nStopping the chatbot.")
        break
    except Exception as e:
        print(f"Erreur : {e}")