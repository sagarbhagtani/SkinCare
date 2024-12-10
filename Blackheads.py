

import cv2


ResDictionary={"class": "Blackheads",
      "severity_Level": "",
      "Marked_image": "",
      "suggested_product": [],
      "count": 0}
def initiateBlackheads(ImageLocations,id):
    print("HelloBH")


    # Step 1: Read the image using OpenCV
    image = cv2.imread("xxyz.jpg")  # Replace with your image file path

    # Resize the image to make it manageable
    image = cv2.resize(image, (540, 740))

    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Create a SimpleBlobDetector with parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10  # Adjust based on dot size

    # Initialize the detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Step 4: Detect blobs (dots)
    keypoints = detector.detect(gray)

    # Step 5: Draw detected blobs as red circles on the original image
    output_image = cv2.drawKeypoints(
        image, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Count the number of dots
    dot_count = len(keypoints)
    ResDictionary = {"class": "Blackheads",
                     "severity_Level": dot_count,
                     "Marked_image": "C:/project/Blackheads/Blackhead"+ id + ".png",
                     "suggested_product": [],
                     "count": 0}





    print(f"Number of black dots detected: {dot_count}")
    output_path = r"C:\project\Blackheads\Blackhead" + id + ".png"
    cv2.imwrite(output_path, output_image)

    return ResDictionary



    # Step 6: Display the result


    # Wait for a key press and close the display window

