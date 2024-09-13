from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import TFSMLayer
import tensorflow as tf
import cv2
import numpy
import pickle
import face_recognition
import imutils
import numpy as np
import dlib
from scipy.spatial import distance as dist
import datetime
import os
import fnmatch2
import sqlite3


'''def identify1(frame, name):
        timestamp = datetime.datetime.now(tz=timezone.utc)
        print(name, timestamp)
        
        path = f"D:/MY_Projects/SAS/attendance/detected/{name}_{timestamp}.png"
        cv2.imwrite(path, frame)
'''


# Load the SavedModel from the directory
#model = tf.keras.models.load_model("D:/Desktop/SAS/check.model")
# Load the SavedMod
# Load the model using TFSMLayer for inference
# model = TFSMLayer("D:/Desktop/SAS", call_endpoint='serving_default')
#model = tf.keras.models.load_model("D:/Desktop/SAS/saved_model.pb")
# model = tf.saved_model.load("D:/Desktop/SAS")
#
# le = pickle.loads(open("le.pickel", "rb").read())
#
#


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_real_vs_fake_model(input_shape=(32, 32, 3)):
    model = Sequential()

    # First Convolutional Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer (Binary Classification)
    model.add(Dense(2, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create the model
real_fake_model = create_real_vs_fake_model()

# Summary of the model
# real_fake_model.summary()

# Load TensorFlow SavedModel
try:
    model =  real_fake_model
except Exception as e:
    print(f"Error loading model: {e}")


net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
# Load the label encoder
le = pickle.loads(open("le.pickel", "rb").read())



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
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=4)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]







def eye_aspect_ratio(eye):

    # Ensure eye is a list of tuples/lists
    eye = np.array(eye)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear



def identify_faces(video_capture,branch,sub):
    face_det = []
    x = 0
    process_this_frame = True
    FULL_POINTS = list(range(0, 68))
    FACE_POINTS = list(range(17, 68))

    JAWLINE_POINTS = list(range(0, 17))
    RIGHT_EYEBROW_POINTS = list(range(17, 22))
    LEFT_EYEBROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 36))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_OUTLINE_POINTS = list(range(48, 61))
    MOUTH_INNER_POINTS = list(range(61, 68))

    EYE_AR_THRESH = 0.30
    EYE_AR_CONSEC_FRAMES = 2

    COUNTER_LEFT = 0
    TOTAL_LEFT = 0

    COUNTER_RIGHT = 0
    TOTAL_RIGHT = 0
    # loading the predictor for predicting
    detector = dlib.get_frontal_face_detector()

    # accessing the shape predictor
    predictor = dlib.shape_predictor("shape_predictor.dat")

    while(True):
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            frame = imutils.resize(frame, width=600)
            for rect in rects:

                x = rect.left()
                y = rect.top()
                x1 = rect.right()
                y1 = rect.bottom()
                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
                left_eye = landmarks[LEFT_EYE_POINTS]
                right_eye = landmarks[RIGHT_EYE_POINTS]
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)
                # calculating blink wheneer the ear value drops down below the threshold
                if ear_left < EYE_AR_THRESH:
                    COUNTER_LEFT += 1
                else:
                    if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                        TOTAL_LEFT += 1
                        COUNTER_LEFT = 0
                if ear_right < EYE_AR_THRESH:
                    COUNTER_RIGHT += 1
                else:
                    if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
                        TOTAL_RIGHT += 1
                        COUNTER_RIGHT = 0
                x = TOTAL_LEFT + TOTAL_RIGHT
        (h, w) = frame.shape[:2]
        temp = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(temp)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # satisfying the union need of verifying through ROI and blink detection.
            if confidence > 0.5 and x > 10:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                preds = model.predict(face)[0]
                # print("pankil")
                # print(preds)
                j = np.argmax(preds)
                # print(le.classes_)
                # print(j)
                if(j==0):
                    j=1
                label = le.classes_[j]
                rorf = label
                label = "{}: {:.4f}".format(label, preds[j])
                if (rorf == "real"):
                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])

                    if process_this_frame:
                        predictions = predict(rgb_frame, model_path="D:/Desktop/SAS/models/trained_model.clf")
                        # print(predictions)

                    process_this_frame = not process_this_frame

                    for name, (top, right, bottom, left) in predictions:
                        if name != "unknown":
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            # Draw a box around the face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                            # # Draw a label with a name below the face
                            # cv2.rectangle(frame, (left, bottom - 35), (right+10, bottom), (0, 255, 0), cv2.FILLED)
                            # font = cv2.FONT_HERSHEY_DUPLEX
                            # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                            # Define the font and scale for the text
                            font = cv2.FONT_HERSHEY_DUPLEX
                            font_scale = 1.0
                            font_thickness = 1

                            # Get the size of the text (width and height)
                            (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale,
                                                                                  font_thickness)

                            # Adjust the right boundary of the rectangle to accommodate the text
                            cv2.rectangle(frame, (left, bottom - text_height - 10), (left + text_width + 10, bottom),
                                          (0, 255, 0), cv2.FILLED)

                            # Place the text on top of the filled rectangle
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, font_scale, (255, 255, 255),
                                        font_thickness)
                            pat = name + '*'
                            files = os.listdir(f'D:/Desktop/SAS/dataset/{branch}')
                            for file in files:
                                if fnmatch2.fnmatch2(file, pat):
                                    roll = file[len(name) + 1:]
                                    break
                            branch = str(branch)
                            sub = str(sub)
                            table = branch + sub
                            now = datetime.datetime.now()
                            date = now.strftime("%d-%m-%Y")
                            time = now.strftime("%I:%M:%S")
                            conn = sqlite3.connect('attendance.db')
                            c = conn.cursor()
                            c.execute(" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='" + table + "'")
                            if c.fetchone()[0] != 1:
                                c.execute(" CREATE TABLE '" + table + "' (name text,roll text,date text,time text)")
                                conn.commit()
                            if name not in face_det:
                                c.execute(
                                    " INSERT INTO '" + table + "' (name,roll,date,time) VALUES ('" + name + "' , '" + roll + "','" + date + "','" + time + "') ")
                                face_det.append(name)
                            conn.commit()
                            conn.close()
                else:
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # print(buf)
        frame = cv2.imencode('.jpg',frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # Display the resulting image

        # cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

