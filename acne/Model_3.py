import ultralytics

from PIL import Image


ResDictionary={"class": "Acne",
      "severity_Level": "",
      "Marked_image": "",
      "suggested_product": [],
      "count": 0}

def initiateAcne(ImageLocations,id):
    print("HelloAC")

    def yolo_Code():
        # Load the model
        from ultralytics import YOLO
        model = YOLO("acne.pt")
        # Load the image you want to test
        image_path = "acne123.jpg"
        image = Image.open(image_path)

        # Perform inference and set a custom save directory
        results = model.predict(image_path, save=True, save_dir="/")


        # Print out the results
        print(results)
        # Iterate through results
        count=0
        for result in results:
            boxes = result.boxes.xyxy  # Bounding boxes (x_min, y_min, x_max, y_max)
            confidences = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Class IDs

            # Example: Print details of each detection
            for i in range(len(boxes)):
                count+=1
                print(f"Box: {boxes[i]}, Confidence: {confidences[i]}, Class: {classes[i]}")

        print(count)
        threshold=0
        import cv2

        img = cv2.imread(image_path)

        for box, cls, conf in zip(boxes, classes, confidences):
            if conf > threshold:
                x1, y1, x2, y2 = map(int, box)
                label = f"Class: {int(cls)}, Conf: {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_image=img

        cv2.imwrite("output_image.jpg", img)
        output_path = r"C:\project\Acne\acne" + id + ".png"
        cv2.imwrite(output_path, output_image)
        ResDictionary = {"class": "Acne",
                         "severity_Level": ""+str(count),
                         "Marked_image": "C:/project/Acne/acne"+id+".png",
                         "suggested_product": [],
                         "count": 0}


        print(ResDictionary)
        return ResDictionary

    return yolo_Code()


def sagar():
    print("Bhagtani's")

