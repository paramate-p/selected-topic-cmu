import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def summarize_text_chatgpt(long_text_script: str, api_key=api_key) -> str:
    """
    Summarizes the provided text using the ChatGPT API.
    
    Args:
        long_text_script (str): The text to be summarized.
    
    Returns:
        str: The generated summary.
    """
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with a valid model available to you
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in text summarization. "
                    "Provide a concise and accurate summary of the text provided."
                )
            },
            {
                "role": "user",
                "content": long_text_script
            }
        ]
    )
    return completion.choices[0].message.content

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes audio to text using the Whisper API.
    
    Args:
        file_path (str): Path to the audio file.
    
    Returns:
        str: The transcribed text.
    """
    with open(file_path, "rb") as audio_file:
        client = OpenAI(api_key=api_key)
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript_response.text


if __name__ == "__main__":
    text = transcribe_audio("../videoplayback.mp4")
    print(text)
