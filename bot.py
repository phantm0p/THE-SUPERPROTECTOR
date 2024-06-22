from pyrogram import Client, filters
from pyrogram.types import Message, Photo
from pymongo import MongoClient
import config
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

# Initialize the MongoDB client
client = MongoClient(config.MONGO_URL)
db = client['telegram_bot']  # Select the database
collection = db['deleted_images']  # Select the collection

# Initialize the bot client
bot = Client("my_bot", api_id=config.api_id, api_hash=config.api_hash, bot_token=config.bot_token)

# Paths to your reference ALLEN logo images
# Get the directory where the script is located
script_directory = os.path.dirname(__file__)
logo_directory = os.path.join(script_directory, "")

# Filter logo files based on naming convention
ALLEN_LOGO_PATHS = [
    os.path.join(logo_directory, filename)
    for filename in os.listdir(logo_directory)
    if filename.startswith("al") and filename.endswith(".png")
]

@bot.on_message(filters.chat(-1002069412308) & filters.photo)
async def handle_photo(client: Client, message: Message):
    try:
        # Check if the image is a Photo type
        if isinstance(message.photo, Photo):
            # Print photo details
            print("Photo received:")
            print(f"  File ID: {message.photo.file_id}")
            print(f"  File Size: {message.photo.file_size}")
            print(f"  Width: {message.photo.width}")
            print(f"  Height: {message.photo.height}")

            # Download the photo
            photo_path = await client.download_media(message.photo.file_id)

            try:
                # Load the downloaded photo
                image = cv2.imread(photo_path)
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                match_found = False
                for logo_path in ALLEN_LOGO_PATHS:
                    # Load the reference ALLEN logo image
                    allen_logo = cv2.imread(logo_path)
                    if allen_logo is None:
                        print(f"Error loading logo image from path: {logo_path}")
                        continue
                    allen_logo_gray = cv2.cvtColor(allen_logo, cv2.COLOR_BGR2GRAY)

                    # Define a range of logo sizes to search for
                    logo_sizes = [
                        (int(allen_logo.shape[1] * scale), int(allen_logo.shape[0] * scale))
                        for scale in [0.5, 0.75, 1.0, 1.25]  # Vary scales as needed
                    ]

                    for logo_width, logo_height in logo_sizes:
                        if logo_width > image_gray.shape[1] or logo_height > image_gray.shape[0]:
                            print(f"Skipping logo size {logo_width}x{logo_height} as it is larger than the target image.")
                            continue

                        # Resize the ALLEN logo
                        allen_logo_resized = cv2.resize(allen_logo_gray, (logo_width, logo_height))

                        # Perform template matching
                        result = cv2.matchTemplate(image_gray, allen_logo_resized, cv2.TM_CCOEFF_NORMED)
                        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                        # Adjust this threshold based on your needs
                        similarity_threshold = 0.8

                        if maxVal > similarity_threshold:
                            await message.delete()
                            await message.reply("Copyright Haa ? , Chutiye Gand Mara...")
                            
                            # Log the deletion in MongoDB
                            collection.insert_one({
                                "user_id": message.from_user.id,
                                "username": message.from_user.username,
                                "file_id": message.photo.file_id,
                                "file_size": message.photo.file_size,
                                "date": datetime.utcnow(),
                                "reason": f"ALLEN logo detected from {logo_path}"
                            })

                            print(f"Image with ALLEN logo (size {logo_width}x{logo_height}) detected and deleted.")
                            match_found = True
                            break  # Stop checking after a match is found

                    if match_found:
                        break  # Stop checking other logos after a match is found

                if not match_found:
                    print("No matching ALLEN logo found in the image.")
            
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                await message.reply("An error occurred while processing your image. Please try again later.")
            except Exception as e:
                print(f"Error processing image: {e}")
                await message.reply("An unexpected error occurred. Please try again later.")
            finally:
                # Clean up the downloaded photo
                os.remove(photo_path)
    except Exception as e:
        print(f"Error handling image message: {e}")
        await message.reply("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    print("Bot is running...")
    bot.run()
