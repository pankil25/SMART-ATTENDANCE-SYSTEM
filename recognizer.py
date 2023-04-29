
import cv2
import pickle
import face_recognition
import datetime
from django.utils import timezone
import xlwrite
import os
import fnmatch2
sub = input('Enter Subject name : ')
batch = input('Enter Batch      : ')
def identify1(frame, name):
        timestamp = datetime.datetime.now(tz=timezone.utc)
        print(name, timestamp)
        path = f"C:/Users/panki/Desktop/attendance/detected/{name}_{timestamp}.png"
        cv2.imwrite(path, frame)




def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def identify_faces(video_capture):


    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="C:/Users/panki/Desktop/attendance/models/trained_model.clf")
            #print(predictions)

        process_this_frame = not process_this_frame

        face_names = []

        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            pat = name+'*'
            files = os.listdir('C:/Users/panki/Desktop/attendance/dataset')
            for file in files:
                if fnmatch2.fnmatch2(file,pat):
                    roll = file[len(name)+1:]
                    break

            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M:%S")
            identify1(frame, name)
            xlwrite.output(sub,batch,name,roll,current_time)

            face_names.append(name)


        # print(buf)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

vid_cam = cv2.VideoCapture(0)
identify_faces(vid_cam)