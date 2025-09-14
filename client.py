

from openai import OpenAI
import os

# set environment variable
os.environ["OPENAI_API_KEY"] = "password"

client = OpenAI(
    base_url="http://localhost:3000/v1"
)
audio = client.audio.speech.create(
    input="你好世界,这里是一个测试,补充维生素c有助于增强免疫力",
    voice="default",
    response_format="wav",
    model="tts-1",
)

temp_file = "temp.wav"
audio.stream_to_file(temp_file)