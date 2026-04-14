from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

import os

load_dotenv()
image_url = r"C:\Users\ossam\Desktop\Desktop\University\sem 8\AI\project\demo_pics\fire_alarm.jpg"
print("Pictue exists: ",os.path.exists(image_url))


client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["API"],
)

org_Model = "object-detection-ojs42/1"
Model2 = "object-detection-ojs42/3"


results = client.infer(image_url, model_id=org_Model)
print(results)

font = ImageFont.truetype("arial.ttf", 40)

img = Image.open(image_url)
draw = ImageDraw.Draw(img)

for pred in results["predictions"]:
    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2

    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    draw.text((x1, y1 - 40), f"{pred['class']}   {pred['confidence']:.0%}", fill="red", font=font)

img.show()

