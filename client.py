

from openai import OpenAI
import os

# set environment variable
os.environ["OPENAI_API_KEY"] = "password"

client = OpenAI(
    base_url="http://localhost:3000/v1"
)
temp_file = "temp.wav"
with client.audio.speech.with_streaming_response.create(
    input="Hello world！这里有全角感叹号，全角逗号",
    voice="dragon",
    response_format="wav",
    model="tts-1",
) as audio:
    audio.stream_to_file(temp_file)