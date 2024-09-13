import cv2
import os
import face_recognition
def take_pic(branch,roll,name):

    img_counter = 0
    DIR = f"D:/Desktop/SAS/dataset/{branch}"
    DIR2 = f"D:/Desktop/SAS/dataset/{branch}/{name}_{roll}"
    try:
        os.mkdir(DIR)
        print("Directory ", branch, " Created ")

    except FileExistsError:
        print("Directory ", branch, " already exists")
        try:
            os.mkdir(DIR2)
            print("Directory ",name," Created")
        except FileExistsError:
            print("Directory ", name, " already exists")
            img_counter = len(os.listdir(DIR2))

    vid_cam = cv2.VideoCapture(0)

    while True:
        ret, frame = vid_cam.read()
        # rgb_small_frame = frame[:, :, ::-1]

        # face_locations = face_recognition.face_locations(rgb_small_frame)
        # print(len(face_locations))

        cv2.imshow("Video", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"image_{img_counter}.jpg"
            # Create the directory if it doesn't exist
            if not os.path.exists(DIR2):
                os.makedirs(DIR2)
                print(f"Directory created: {DIR2}")

            success = cv2.imwrite(os.path.join(DIR2, img_name), frame)
            if success:
                print(f"{img_name} captured and saved successfully!")
            else:
                print(f"Error: {img_name} could not be saved!")

            print("{} Captured..!".format(img_name))
            img_counter += 1

    vid_cam.release()

    cv2.destroyAllWindows()
    return img_counter