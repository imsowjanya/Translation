import os
import pandas as pd
import pycld2 as cld2
import torch
from flask import Flask, request, jsonify
from transformers import SeamlessM4Tv2Model, AutoProcessor

app = Flask(__name__)

# Load API key from environment variable
API_KEY = "cloud9898"

# Function to detect language using pycld2
def detect_language(text):
    is_reliable, _, details = cld2.detect(text)
    return details[0][1] if is_reliable else None  # Returning the language code of the most confidently detected language

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

# Language mapping
lang_mapping = {"te": "tel", "kn": "kan", "ta": "tam", "ma": "mal", "hi": "hin", "mar": "mar", "gu": "guj"}

# Check API key
def check_api_key(api_key):
    # Example logic: You can check if the provided API key matches the expected key
    return api_key == API_KEY

@app.route('/translate-text', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    api_key = request.headers.get("API-Key")
    if not check_api_key(api_key):
        return jsonify({'error': 'Invalid API key'}), 401

    original_text = data['text']
    detected_lang = detect_language(original_text)
    detected_lang = lang_mapping.get(detected_lang, detected_lang)
    
    # Process input
    text_inputs = processor(text=original_text, src_lang=detected_lang, return_tensors="pt").to(device)
    
    # Generate translation
    output_tokens = model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
    
    # Pass the tensor directly to the decode method
    translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    
    return jsonify({'original_text': original_text, 'detected_language': detected_lang, 'translated_text': translated_text})

@app.route('/translate-file', methods=['POST'])
def translate_file():
    api_key = request.headers.get("API-Key")
    if not check_api_key(api_key):
        return jsonify({'error': 'Invalid API key'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    input_df = pd.read_csv(file)
    
    # Detect language for each text entry
    input_df["Detected_Language"] = input_df["Text"].apply(detect_language)
    input_df["Detected_Language"] = input_df["Detected_Language"].map(lang_mapping).fillna(input_df["Detected_Language"])
    
    # Translate each text entry and store the translation along with the original text in a new DataFrame
    output_df = pd.DataFrame(columns=["Original_Text", "Detected_Language", "Translated_Text"])

    for index, row in input_df.iterrows():
        original_text = row["Text"]
        detected_lang = row["Detected_Language"]
        
        # Process input
        text_inputs = processor(text=original_text, src_lang=detected_lang, return_tensors="pt").to(device)
        
        # Generate translation
        output_tokens = model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
        
        # Pass the tensor directly to the decode method
        translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        
        # Store the translation along with the original text in the DataFrame
        temp_df = pd.DataFrame({"Original_Text": [original_text], "Detected_Language": [detected_lang], "Translated_Text": [translated_text]})
        output_df = pd.concat([output_df, temp_df], ignore_index=True)
    
    result = output_df.to_dict(orient='records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4545)
