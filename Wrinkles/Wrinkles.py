from skimage.filters import frangi, gabor
from skimage import measure, morphology
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

severity_Level=0
ResDictionary={"class": "Wrinkles",
      "severity_Level": "",
      "Marked_image": "C:\project\Wrinkles",
      "suggested_product": [],
      "count": 0}
def initiateWrinkles(ImageLocations,id):
    print("HelloWR")



    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


    forehead_landmarks = [389, 368, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70, 139, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
    right_cheek_landmarks = [340, 280, 425, 423, 371, 349, 340]
    left_cheek_landmarks = [121, 47, 142, 203, 205, 123, 116, 111, 121]


    def get_face_region(image, landmarks, region_landmarks):
        h, w, _ = image.shape
        # Get the region points based on the landmark indices
        region_points = [(int(landmarks[pt].x * w), int(landmarks[pt].y * h)) for pt in region_landmarks]

        # Create a mask with the same dimensions as the image
        mask = np.zeros((h, w), dtype=np.uint8)

        # Fill the polygon defined by the region points on the mask
        cv2.fillPoly(mask, [np.array(region_points, dtype=np.int32)], 255)

        # Create a new image with an alpha channel (transparent background)
        transparent_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Set the alpha channel to 0 (transparent) where mask is 0 (background), else keep original colors
        transparent_image[:, :, 3] = mask

        # Crop the ROI from the transparent image (optional, if you want only the bounding box region)
        x, y, w, h = cv2.boundingRect(np.array(region_points, dtype=np.int32))
        face_region = transparent_image[y:y+h, x:x+w]

        return face_region
    def overlay_wrinkles(original_image, wrinkle_map, region_landmarks, landmarks):
        h, w, _ = original_image.shape
        # Convert the region landmarks to points
        region_points = [(int(landmarks[pt].x * w), int(landmarks[pt].y * h)) for pt in region_landmarks]

        # Scale wrinkle map to fit the region
        x, y, region_w, region_h = cv2.boundingRect(np.array(region_points, dtype=np.int32))
        wrinkle_map_resized = cv2.resize(wrinkle_map.astype('uint8'), (region_w, region_h), interpolation=cv2.INTER_NEAREST)

        # Create a blank overlay for drawing wrinkles
        wrinkle_overlay = np.zeros_like(original_image)

        # Add wrinkles in red color
        wrinkle_overlay[y:y+region_h, x:x+region_w, 1] = wrinkle_map_resized * 255

        # Combine with the original image
        result = cv2.addWeighted(original_image, 0.8, wrinkle_overlay, 4, 0)
        return result



    def master_control(image):
        # # image = cv2.resize(image, (int(image.shape[1]*0.3), int(image.shape[0]*0.3)), interpolation=cv2.INTER_CUBIC)  # 图片分辨率很大是要变小
        # b, g, r = cv2.split(image)  # image
        if len(image.shape) == 3 and image.shape[2] == 3:  # Check for 3 channels (RGB or BGR)
            b, g, r = cv2.split(image)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # Check for 4 channels (RGBA)
            b, g, r, a = cv2.split(image)
        else:
            # Handle the case for grayscale or other formats
            print("Error: Unsupported image format")
            return

        #sk_frangi_img = frangi(g, scale_range=(0, 1), scale_step=0.01, beta1=1.5, beta2=0.01)  # 线宽范围，步长，连接程度（越大连接越多），减少程度(越大减得越多)0.015
        sk_frangi_img = frangi(g, scale_range=(1, 10), scale_step=2, beta=0.5, alpha=0.5, gamma=15)

        sk_frangi_img = morphology.closing(sk_frangi_img, morphology.disk(1))
        sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency=0.35, theta=0)
        sk_gabor_img_2, sk_gabor_2 = gabor(g, frequency=0.35, theta=45)  # 越小越明显 the smaller it is
        sk_gabor_img_3, sk_gabor_3 = gabor(g, frequency=0.35, theta=90)
        sk_gabor_img_4, sk_gabor_4 = gabor(g, frequency=0.35, theta=360)  # 横向皱纹
        sk_gabor_img_1 = morphology.opening(sk_gabor_img_1, morphology.disk(2))
        sk_gabor_img_2 = morphology.opening(sk_gabor_img_2, morphology.disk(1))
        sk_gabor_img_3 = morphology.opening(sk_gabor_img_3, morphology.disk(2))
        sk_gabor_img_4 = morphology.opening(sk_gabor_img_4, morphology.disk(2))
        all_img = cv2.add(0.1 * sk_gabor_img_2, 0.9 * sk_frangi_img)  # + 0.02 * sk_gabor_img_1 + 0.02 * sk_gabor_img_2 + 0.02 * sk_gabor_img_3
        all_img = morphology.closing(all_img, morphology.disk(1))
        _, all_img = cv2.threshold(all_img, 0.3, 1, 0)
        img1 = all_img
        # print(all_img, all_img.shape, type(all_img))
        # contours, image_cont = cv2.findContours(all_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # all_img = all_img + image_cont
        bool_img = all_img.astype(bool)
        label_image = measure.label(bool_img)
        count = 0

        for region in measure.regionprops(label_image):
            if region.area < 10: #   or region.area > 700
                x = region.coords
                for i in range(len(x)):
                    all_img[x[i][0]][x[i][1]] = 0
                continue
            if region.eccentricity > 0.98:
                count += 1
            else:
                x = region.coords
                for i in range(len(x)):
                    all_img[x[i][0]][x[i][1]] = 0

        skel, distance = morphology.medial_axis(all_img.astype(int), return_distance=True)
        skels = morphology.closing(skel, morphology.disk(1))
        trans1 = skels  # 细化
        return skels, count  # np.uint16(skels.astype(int))


    # def face_wrinkle(path):
    #     # result = pa.curve(path, backage)
    #     result = cv2.imread(path)
    #     img, count = master_control(result)
    #     print(img.astype(float))
    #     result[img > 0.1] = 255
    #     cv2.imshow("result", img.astype(float))
    #     cv2.waitKey(0)






    # def display_regions(forehead, left_cheek, right_cheek):
    #     # Create subplots for all three regions
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    #     # Display the forehead region in color
    #     axes[0].imshow(forehead.astype('uint8'))  # Ensure image is in uint8 for display
    #     axes[0].set_title("Forehead Region")
    #     axes[0].axis('off')  # Hide axis
    #
    #     # Display the left cheek region in color
    #     axes[1].imshow(left_cheek.astype('uint8'))  # Ensure image is in uint8 for display
    #     axes[1].set_title("Left Cheek Region")
    #     axes[1].axis('off')  # Hide axis
    #
    #     # Display the right cheek region in color
    #     axes[2].imshow(right_cheek.astype('uint8'))  # Ensure image is in uint8 for display
    #     axes[2].set_title("Right Cheek Region")
    #     axes[2].axis('off')  # Hide axis
    #
    #     # Show the plots
    #     plt.show()



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
                forehead_region = get_face_region(image, face_landmarks.landmark, forehead_landmarks)
                right_cheek_region = get_face_region(image, face_landmarks.landmark, right_cheek_landmarks)
                left_cheek_region = get_face_region(image, face_landmarks.landmark, left_cheek_landmarks)

                # forehead_region=cv2.resize(forehead_region,(200,150))
                # right_cheek_region=cv2.resize(right_cheek_region,(200,150))
                # left_cheek_region=cv2.resize(left_cheek_region,(200,150))


                #display_regions(forehead_region, left_cheek_region, right_cheek_region)

                # Detect wrinkles on extracted regions
                forehead_wrinkles, forehead_count = master_control(forehead_region)
                right_cheek_wrinkles, right_cheek_count = master_control(right_cheek_region)
                left_cheek_wrinkles, left_cheek_count = master_control(left_cheek_region)

                # Overlay wrinkles on the original image
                image_with_wrinkles = overlay_wrinkles(original_image, forehead_wrinkles, forehead_landmarks, face_landmarks.landmark)
                image_with_wrinkles = overlay_wrinkles(image_with_wrinkles, left_cheek_wrinkles, left_cheek_landmarks, face_landmarks.landmark)
                image_with_wrinkles = overlay_wrinkles(image_with_wrinkles, right_cheek_wrinkles, right_cheek_landmarks, face_landmarks.landmark)
                global severity_Level
                output_path = r"C:\project\Wrinkles\Wrinkle"+id+".png"
                cv2.imwrite(output_path, image_with_wrinkles)

                # Display the final image with wrinkles overlay
                severity_Level= forehead_count + left_cheek_count + right_cheek_count
                # cv2.imshow("Detected Wrinkles", image_with_wrinkles)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Print wrinkle counts
                print(f"Forehead wrinkle count: {forehead_count}")
                print(f"Left cheek wrinkle count: {left_cheek_count}")
                print(f"Right cheek wrinkle count: {right_cheek_count}")





    path = r"Wrinkles/wri_09.jpeg"
    process_image(path)
    ResDictionary.update({"class": "Wrinkles",
          "severity_Level": severity_Level,
          "Marked_image": "C:/project/Wrinkles/Wrinkle"+id+".png",
          "suggested_product": [],
          "count": 0})
    print(ResDictionary)

    return ResDictionary
