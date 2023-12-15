
import time
import requests
import base64
from PIL import ImageGrab
import os
import time
from typing import Tuple, List
import os
import base64
import requests
from typing import Tuple
import aiohttp
import asyncio

api_key = os.getenv('OPENAI_API_KEY')



def take_screenshot():
    # Ensure the 'images/' directory exists
    if not os.path.exists('images'):
        os.makedirs('images')

    # Generate a unique filename using the current timestamp
    filename = f"images/screenshot_{int(time.time())}.png"

    # Capture the screenshot
    screenshot = ImageGrab.grab()

    # Save the screenshot
    screenshot.save(filename)

    print(f"Screenshot saved as {filename}")



def see_computer_screen() -> Tuple[bool, str]:
    """
    Takes a screenshot, analyzes it using the OpenAI GPT-4 vision model, and then deletes the screenshot.

    Returns:
    Tuple[bool, str]: A tuple containing a boolean and a string. The boolean is True if the analysis was successful, 
                      False otherwise. The string contains the analysis result or an error message.
    """
    try:
        # Take a screenshot
        take_screenshot()

        # Find the most recent screenshot in the 'images/' directory
        image_path = find_most_recent_image('images')
        if not image_path:
            return False, "No images found in the 'images/' directory."

        # Encode the image
        base64_image = encode_image(image_path)

        # Set up the request headers and payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        # Make the API request
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Delete the screenshot after analysis
        os.remove(image_path)

        return True, response.json()
    except Exception as e:
        # Delete the screenshot in case of an error
        if image_path:
            os.remove(image_path)
        return False, f"An error occurred: {e}"


def find_most_recent_image(directory: str) -> str:
    """
    Finds the most recent image file in a specified directory.

    Args:
    directory (str): The directory to search in.

    Returns:
    str: The path to the most recent image file, or an empty string if no image is found.
    """
    try:
        # List all files in the directory and filter out non-image files
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Sort files by modification time
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        # Return the most recent file
        return files[0] if files else ""
    except Exception as e:
        print(f"Error finding the most recent image: {e}")
        return ""

def encode_image(image_path: str) -> str:
    """
    Encodes an image to a base64 string.

    Args:
    image_path (str): The path to the image file.

    Returns:
    str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')




async def see_computer_screen_async() -> Tuple[bool, str]:
    try:
        # Take a screenshot (this part needs to be adapted if it's IO-bound)
        take_screenshot()

        image_path = find_most_recent_image('images')
        if not image_path:
            return False, "No images found in the 'images/' directory."

        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }

        # Asynchronous HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
                response_data = await response.json()

                # Extract the description text from the response
        description_text = response_data['choices'][0]['message']['content']

        # Delete the screenshot after analysis
        os.remove(image_path)

        return True, description_text
    except Exception as e:
        if image_path:
            os.remove(image_path)
        return False, f"An error occurred: {e}"
