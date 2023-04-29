import cv2
import os
import face_recognition

img_counter = 0

dirName = input("Please enter your name: ").upper()
dirID = input("Please enter ID: ")

DIR = f"C:/Users/panki/Desktop/attendance/dataset/{dirName}_{dirID}"

try:
    os.mkdir(DIR)
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")
    img_counter = len(os.listdir(DIR))

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
        img_name = f"C:/Users/panki/Desktop/attendance/dataset/{dirName}_{dirID}/sample_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print("{} Captured..!".format(img_name))
        img_counter += 1

vid_cam.release()

cv2.destroyAllWindows()
