

from openai import OpenAI
import os

# set environment variable
os.environ["OPENAI_API_KEY"] = "password"

client = OpenAI(
    base_url="http://localhost:3000/v1"
)
audio = client.audio.speech.create(
    input="Hello world！这里有全角感叹号，全角逗号",
    voice="dragon",
    response_format="wav",
    model="tts-1",
)

temp_file = "temp.wav"
audio.stream_to_file(temp_file)