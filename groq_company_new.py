import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
import os
import datetime
from groq import Groq

# Initialize Groq client
client = Groq(
    api_key="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",
)

def get_text_chunks(text, chunk_size=5500):
    """Split text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def clean_and_format_json(response_content):
    # Remove leading and trailing artifacts and strip excess whitespace
    response_content = response_content.lstrip("`").lstrip("json").strip("`").strip()

    # Fix common encoding artifacts
    response_content = response_content.replace("â€™", "'").replace("â€œ", '"').replace("â€�", '"')
    response_content = response_content.replace("Letâ€™s", "let's")

    # Remove any remaining non-ASCII characters
    response_content = re.sub(r'[^\x00-\x7F]+', '', response_content)

    # Ensure response content starts with '[' and ends with ']', truncate if necessary
    if not response_content.startswith("["):
        response_content = "[" + response_content
    if not response_content.endswith("]"):
        last_bracket_position = response_content.rfind("}")
        if last_bracket_position != -1:
            response_content = response_content[:last_bracket_position + 1] + "]"
    
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        # Attempt to truncate and re-parse if incomplete JSON
        last_bracket_position = response_content.rfind("}")
        response_content = response_content[:last_bracket_position + 1] + "]"
        return json.loads(response_content)

def process_chunk_with_retries(client, prompt, retries=3, initial_wait_time=60):
    """Process a single chunk with retry logic on rate limit errors."""
    wait_time = initial_wait_time
    for attempt in range(retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                print(f"Rate limit exceeded. Attempt {attempt + 1} of {retries}. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  # Incremental back-off
            else:
                raise e
    raise Exception("Max retries exceeded on rate limit.")

def get_qa_data_from_text(text_content, company_name, role):
    # Initialize list to hold all Q&A results
    qa_results = []

    # Process each chunk and send to Groq API with delays
    chunks = get_text_chunks(text_content, chunk_size=5500)
    print(f"Number of chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        prompt = f"""Carefully extract all questions and corresponding answers from the following text. 
        Ensure that you extract every possible question-answer pair without omitting any. 
        - Format each question and answer pair as JSON objects in a single JSON array.
        - Use the keys 'question' for the question and 'answer' for the answer.
        - Do not include any additional parameters or characters outside of the JSON array.
        - Ensure that the JSON output does not contain any special encoding artifacts, such as â€™, â€œ, or other non-ASCII characters. Use standard ASCII characters only.
        - If no answer is provided for a question, set the 'answer' value to null.\n\n{chunk}\n\nFormat the output as a JSON array of objects with keys 'question' and 'answer'."""

        try:
            # Use the retry logic for each chunk
            response_content = process_chunk_with_retries(client, prompt)

            # Debugging output for the Groq API response
            print(f"Groq API response content for chunk {i + 1}:", response_content[:500])

            # Clean and format the response content to ensure valid JSON array
            qa_data = clean_and_format_json(response_content)

            # Accumulate results
            qa_results.extend(qa_data if isinstance(qa_data, list) else [qa_data])

            # Delay between chunk requests to avoid rate limiting
            time.sleep(5)

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in chunk {i + 1}: {e}")
            print("Response content causing the error:", response_content)
        
        except Exception as e:
            print(f"An error occurred in chunk {i + 1}: {e}")

    # Convert the accumulated results to a DataFrame and add extra columns
    df = pd.DataFrame(qa_results)
    if not df.empty:
        # Add company and role columns
        df['company_name'] = company_name
        df['role'] = role
        return df
    else:
        print("No Q&A pairs found, returning an empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if no data extracted

def process_and_save_qa_data(text_content, company_name, role, folder_path="csv"):
    # Update folder path to include company name
    folder_path = os.path.join(folder_path, f"{company_name.lower()}_qa_data")

    # Process the Q&A data from the text content
    df = get_qa_data_from_text(text_content, company_name, role)

    if not df.empty:
        # Create folder if it does not exist
        os.makedirs(folder_path, exist_ok=True)

        # File name for appending data
        file_name = f"{company_name.lower()}_qa_data.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            # Append to the file if it already exists
            df.to_csv(file_path, mode='a', index=False, header=False)
        else:
            # Save with headers if the file does not exist
            df.to_csv(file_path, index=False)

        print(f"Data for {company_name} with role {role} saved to {file_path}")
    else:
        print(f"No data to save for {company_name}.")

# Example Usage
text_content = """
"""

company_name = "Discord"
role = "Data Scientist"

process_and_save_qa_data(text_content, company_name, role)
