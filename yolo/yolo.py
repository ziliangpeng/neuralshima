from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

# Load an image from a URL or your local file system
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# url = 'https://cdn.guidedogs.com.au/wp-content/uploads/2021/01/Two-Gold-St-Kilda-610x525-lqip.jpg'
# url = 'https://facts.net/wp-content/uploads/2023/12/11-stop-sign-facts-1701615226.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Initialize feature extractor and model
model_name = "Ultralytics/YOLOv8"
model_name = "hustvl/yolos-base"
model_name = "hustvl/yolos-small"
feature_extractor = YolosFeatureExtractor.from_pretrained(model_name)
model = YolosForObjectDetection.from_pretrained(model_name)
class_labels = model.config.id2label
print(class_labels)

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)

# Extract logits and bounding boxes from outputs
logits = outputs.logits
bboxes = outputs.pred_boxes

# print("Logits:", logits)
# print("Bounding Boxes:", bboxes)
print("logits shape:", logits.shape)
print("Bounding Boxes shape:", bboxes.shape)

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Function to visualize the image and the bounding boxes
def visualize_bboxes(image, bboxes, logits, threshold=0.5):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    # Iterate through the bounding boxes and plot them
    for i in range(len(logits)):
        print(threshold, logits[i].max())
        if logits[i].max() > threshold:  # Ensure high confidence
            print(logits[i])
            predicted_class_id = logits[i].argmax(-1).item()
            print(predicted_class_id)
            if predicted_class_id in class_labels:
                predicted_class_label = class_labels[predicted_class_id]
                print(predicted_class_label)
            bbox = bboxes[i].tolist()
            center_x, center_y, width, height = bbox
            y = center_y - height / 2
            x = center_x - width / 2

            x, y, width, height = (
                x * image.width,
                y * image.height,
                width * image.width,
                height * image.height,
            )

            print(bbox)
            print(image.width, image.height)
            # Convert normalized coordinates to actual pixel values
            # x, y, width, height = bbox[0] * image.width, bbox[1] * image.height, bbox[2] * image.width, bbox[3] * image.height
            print(x, y, width, height)
            rect = patches.Rectangle(
                (x, y),  # (x,y)
                width,  # width
                height,  # height
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

    plt.axis("off")
    plt.show()


# Example usage:
visualize_bboxes(image, bboxes[0], logits[0], threshold=0.4)
