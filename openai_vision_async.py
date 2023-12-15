import asyncio
import json
import time
import openai_vision as eye

async def async_screen_reading_and_storing():
    while True:
        success, description = await eye.see_computer_screen_async()  # Assume this is an async function
        if success:
            timestamp = time.time()
            store_description(description, timestamp)  # Function to store descriptions
        await asyncio.sleep(1)  # Adjust the frequency as needed

def store_description(description, timestamp):
    # Implement logic to store the last three descriptions with timestamps in a text file
    # ...
