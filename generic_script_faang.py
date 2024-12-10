import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from groq import Groq

# Initialize Groq client
client = Groq(
    api_key="",
)

def get_qa_data(url):
    # Step 1: Scrape website HTML
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    html = response.text

    # Step 2: Extract text data from the HTML using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    text_data = [p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])]  # Adjusting selectors to gather relevant content
    text_content = " ".join(text_data)

    # Step 3: Prepare prompt for Groq model to extract Q&A pairs
    prompt = f""" Carefully extract all questions and corresponding answers from the following text. 
    Ensure that you extract every possible question-answer pair without omitting any. 
    - Format each question and answer pair as JSON objects.
    - Use the keys 'question' for the question and 'answer' for the answer.
    - Do not stop until all questions are extracted.
    - Don't give .. and so on; instead, give all questions and answers pairs.
    - Do not include anything in the response except for the JSON array of objects, not even json as a starting word, the string should start with [ and end with ].
    - If no answer is provided for a question, set the 'answer' value to null.\n\n{text_content}\n\nFormat the output as JSON objects with keys 'question' and 'answer'."""

    try:
        # Step 4: Use Groq API to generate a response
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            #model="llama3-8b-8192",  # Replace with the appropriate model
            model="llama-3.1-70b-versatile",
        )

        # Parse the response content as JSON
        response_content = chat_completion.choices[0].message.content
        qa_data = json.loads(response_content)

        # Ensure the extracted data is in list format
        if not isinstance(qa_data, list):
            qa_data = [qa_data]

        # Step 5: Convert the extracted Q&A data to a DataFrame
        df = pd.DataFrame(qa_data)

        return df

    except json.JSONDecodeError as e:
        print("JSONDecodeError:", e)
        print("Response content:", response_content)  # Show the content causing the error
        return pd.DataFrame()  # Return an empty DataFrame on error

    except Exception as e:
        print("An error occurred:", e)
        return pd.DataFrame()  # Return an empty DataFrame on any other error



website_url = 'https://unp.education/content/apple-data-scientist-interview-questions-answers/'
df = get_qa_data(website_url)

# Print the DataFrame to see the output
print(df)

# Save the output to a CSV file
df.to_csv('output.csv', index=False)