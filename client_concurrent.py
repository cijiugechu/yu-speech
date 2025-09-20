import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# Set environment variable (same logic as client.py)
os.environ["OPENAI_API_KEY"] = "password"


def tts_request(index: int, text: str, voice: str, model: str, response_format: str):
    start_perf = time.perf_counter()
    start_wall = datetime.now()
    try:
        client = OpenAI(base_url="http://localhost:3000/v1")
        out_file = f"temp_{index + 1}.wav"
        with client.audio.speech.with_streaming_response.create(
            input=text,
            voice=voice,
            response_format=response_format,
            model=model,
        ) as audio:
            audio.stream_to_file(out_file)
        end_perf = time.perf_counter()
        end_wall = datetime.now()
        return {
            "index": index,
            "ok": True,
            "file": out_file,
            "text": text,
            "start": start_wall,
            "end": end_wall,
            "duration": end_perf - start_perf,
        }
    except Exception as e:
        end_perf = time.perf_counter()
        end_wall = datetime.now()
        return {
            "index": index,
            "ok": False,
            "error": str(e),
            "text": text,
            "start": start_wall,
            "end": end_wall,
            "duration": end_perf - start_perf,
        }


def main():
    texts = [
        "Hello world! 这是一条英文混合中文的句子。",
        "第二条请求，测试并发处理能力。",
        "第三条请求，包含一些标点符号！？，。",
        "第四条请求，用于压力测试。",
        "第五条请求，完成并记录耗时。",
    ]

    voice = "dragon"
    model = "tts-1"
    response_format = "wav"

    print(f"Starting {len(texts)} concurrent requests...\n")

    futures = []
    with ThreadPoolExecutor(max_workers=len(texts)) as executor:
        for i, text in enumerate(texts):
            futures.append(executor.submit(tts_request, i, text, voice, model, response_format))

        for future in as_completed(futures):
            result = future.result()
            idx = result["index"] + 1
            start_str = result["start"].strftime("%H:%M:%S")
            end_str = result["end"].strftime("%H:%M:%S")
            if result["ok"]:
                print(
                    f"[#{idx}] start={start_str}, end={end_str}, duration={result['duration']:.2f}s, file={result['file']}"
                )
            else:
                print(
                    f"[#{idx}] start={start_str}, end={end_str}, duration={result['duration']:.2f}s, ERROR: {result['error']}"
                )


if __name__ == "__main__":
    main()


