
# print(output_df)
import pandas as pd
import pycld2 as cld2
import torch
from transformers import SeamlessM4Tv2Model, AutoProcessor

# Function to detect language using pycld2
def detect_language(text):
    is_reliable, _, details = cld2.detect(text)
    return details[0][1] if is_reliable else None  # Returning the language code of the most confidently detected language

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

# Read input from CSV file
# input_csv_file = "/root/navneetha/Translation/cleaned_text.csv"
input_csv_file = "/home/swj/Downloads/navneetha/Translation/cleaned_text.csv"
output_csv_file = "output.csv"

input_df = pd.read_csv(input_csv_file)

# Detect language for each text entry
input_df["Detected_Language"] = input_df["Text"].apply(detect_language)

# Map detected languages to compatible language codes
lang_mapping = {"te": "tel", "kn": "kan", "ta": "tam", "ma": "mal", "hi": "hin", "mar": "mar", "gu": "guj"}
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

# Save the output DataFrame to a new CSV file
output_df.to_csv(output_csv_file, index=False)

print(output_df)
