import cv2
import mediapipe as mp
import numpy as np

ResDictionary={"class": "OilySkin",
      "severity_Level": "",
      "Marked_image": "",
      "suggested_product": [],
      "count": 0}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def initiateOilySkin(ImageLocations,id):

      print("Hello_OilySkin")
      mp_face_mesh = mp.solutions.face_mesh
      face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

      forehead_landmarks = [389, 368, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70, 139, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
      right_cheek_landmarks = [340, 280, 425, 423, 371, 349, 340]
      left_cheek_landmarks = [121, 47, 142, 203, 205, 123, 116, 111, 121]


      def detect_oily_skin(image, mask):
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
          lower_skin = np.array([0, 20, 70], dtype=np.uint8)
          upper_skin = np.array([20, 255, 255], dtype=np.uint8)
          skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
          shine_mask = cv2.inRange(gray, 190, 255)
          oily_skin_mask = cv2.bitwise_and(skin_mask, shine_mask)
          oily_skin_region = cv2.bitwise_and(oily_skin_mask, mask)
          return oily_skin_region


      def get_face_region(image, landmarks, region_landmarks):
          h, w, _ = image.shape
          region_points = [(int(landmarks[pt].x * w), int(landmarks[pt].y * h)) for pt in region_landmarks]
          return region_points


      def process_image(image_path):
          image = cv2.imread(image_path)
          if image is None:
              print("Error: Could not load image. Please check the file path.")
              return
          rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          original_image = image.copy()

          # Initialize face mesh results
          with face_mesh as fm:
              results = fm.process(rgb_image)
              if results.multi_face_landmarks:
                  face_landmarks = results.multi_face_landmarks[0]

                  # Extract regions for forehead, right cheek, and left cheek
                  forehead_points = get_face_region(image, face_landmarks.landmark, forehead_landmarks)
                  right_cheek_points = get_face_region(image, face_landmarks.landmark, right_cheek_landmarks)
                  left_cheek_points = get_face_region(image, face_landmarks.landmark, left_cheek_landmarks)

                  # Create masks for the regions
                  mask_forehead = np.zeros(image.shape[:2], dtype=np.uint8)
                  mask_right_cheek = np.zeros(image.shape[:2], dtype=np.uint8)
                  mask_left_cheek = np.zeros(image.shape[:2], dtype=np.uint8)

                  cv2.fillPoly(mask_forehead, [np.array(forehead_points, dtype=np.int32)], 255)
                  cv2.fillPoly(mask_right_cheek, [np.array(right_cheek_points, dtype=np.int32)], 255)
                  cv2.fillPoly(mask_left_cheek, [np.array(left_cheek_points, dtype=np.int32)], 255)

                  # Detect oily skin in each region
                  forehead_oily = detect_oily_skin(image, mask_forehead)
                  right_cheek_oily = detect_oily_skin(image, mask_right_cheek)
                  left_cheek_oily = detect_oily_skin(image, mask_left_cheek)

                  # Count white pixels
                  forehead_white_pixels = np.count_nonzero(forehead_oily)
                  right_cheek_white_pixels = np.count_nonzero(right_cheek_oily)
                  left_cheek_white_pixels = np.count_nonzero(left_cheek_oily)

                  print(f"Forehead oily area (white pixels): {forehead_white_pixels}")
                  print(f"Right cheek oily area (white pixels): {right_cheek_white_pixels}")
                  print(f"Left cheek oily area (white pixels): {left_cheek_white_pixels}")

                  # Overlay results on the original image
                  overlay_image = original_image.copy()
                  overlay_image[forehead_oily > 0] = [0, 0, 255]  # Red for forehead
                  overlay_image[right_cheek_oily > 0] = [0, 255, 0]  # Green for right cheek
                  overlay_image[left_cheek_oily > 0] = [255, 0, 0]  # Blue for left cheek
                  output_path = r"C:\project\OilySkin\oilySkin" + id + ".png"
                  cv2.imwrite(output_path, overlay_image)

                  # Display results
                  # cv2.imshow("Original Image", original_image)
                  # cv2.imshow("Oily Forehead", forehead_oily)
                  # cv2.imshow("Oily Right Cheek", right_cheek_oily)
                  # cv2.imshow("Oily Left Cheek", left_cheek_oily)
                  overlay_image=cv2.resize(overlay_image,(640,480))
                  # cv2.imshow("Overlay Image", overlay_image)
                  # cv2.waitKey(0)
                  # cv2.destroyAllWindows()
                  ResDictionary = {"class": "OilySkin",
                                   "severity_Level": "",
                                   "Marked_image": "C:/project/OilySkin/oilySkin" + id + ".png",
                                   "suggested_product": [],
                                   "count": forehead_white_pixels+right_cheek_white_pixels+left_cheek_white_pixels}
          return ResDictionary


      return process_image("OilySkin/oily641.webp")



