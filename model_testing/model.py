from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import os

#to load the env api key
load_dotenv()
#folder path for the pics used for testing
folder_path = r"C:\Users\ossam\Desktop\Desktop\University\sem 8\AI\project\demo_pics"
print("Pictue exists: ",os.path.exists(folder_path))



org_Model = "object-detection-ojs42/1"
Model2 = "object-detection-ojs42/3"


#the trained model
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["API"],
)

#font for the predicted text
font = ImageFont.truetype("arial.ttf", 30)

#loop through the pictures in the folder and detect the obejct in each
for image_path in os.listdir(folder_path):
        print(image_path)

        image = os.path.join(folder_path, image_path)
        results = client.infer(image, model_id=org_Model)
        print(results)

        img = Image.open(image).convert("RGB")
        draw = ImageDraw.Draw(img)

        for pred in results["predictions"]:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2

            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            draw.text((x1, y1 - 35), f"{pred['class']}   {pred['confidence']:.0%}", fill="red", font=font)

        img.show()
