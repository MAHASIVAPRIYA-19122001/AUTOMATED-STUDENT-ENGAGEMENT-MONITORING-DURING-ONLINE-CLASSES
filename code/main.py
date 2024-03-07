import cv2
import os
cap = cv2.VideoCapture("D:\\utube\\utube\\student19 ancy.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frames_per_minute = int(fps * 2.4)
minute_count = 0
frame_count = 0
frames_folder = "D:\\learning_frames_faces_eyes\\own_dataset_frames\\ANCY\\"
os.makedirs(frames_folder, exist_ok=True)
output_folder_eyes =
"D:\\learning_frames_faces_eyes\\survey_eyes\\survey_eyes\\ANCY\\"
output_folder_faces="D:\learning_frames_faces_eyes\\survey_faces\\survey_faces\\ANCY
\\"
os.makedirs(output_folder_faces, exist_ok=True)
os.makedirs(output_folder_eyes, exist_ok=True)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_defa
ult.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#focus detection CNN architecture
from keras.models import load_model
model = load_model("C:\\Users\\DELL\\Downloads\\code\\f1.h5")
# Evaluate the model on the testing data
score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#testing with sequence of frames
60
folder_path = "D:\\learning_frames_faces_eyes\\survey_eyes\\survey_eyes\\ANCY\\"
class_names = ['focused', 'not-focused']
# Loop over each image file in the folder
predicted_class = np.argmax(result)
print(f'Predicted class: {predicted_class}')
# Print the predicted class
if predicted_class == 0:
print("focused")
detect_face(0)
else:
print("not-focused")
detect_face(1)
#Dominant emotion and emotion probability using mobilenet
# Load the pre-trained model
model = load_model("C:\\Users\\DELL\\Downloads\\code\\new.h5")
folder_path = "D:\\quiz_faces\\fmadhu\\"
emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# Find the mode emotion for the subfolder
mode_emotions = [emotions[i] for i in y_preds]
mode_emotion = mode(mode_emotions)
mode_emotion_probability = max(emotion_probabilities[mode_emotion])
mode_emotion_probabilities.append(mode_emotion_probability)
mode_emotion_index = emotions.index(mode_emotion)
mode_emotion_indices.append(mode_emotion_index)
print(f"{subfolder}: Mode emotion: {mode_emotion}, Mode emotion probability:
{mode_emotion_probability}")
if mode_emotion=='angry':
detect_faces('angry')
elif mode_emotion=='disgust':
detect_faces('disgust')
elif mode_emotion=='fear':
detect_faces('fear')
61
elif mode_emotion=='happy':
detect_faces('happy')
elif mode_emotion=='neutral':
detect_faces('neutral')
elif mode_emotion=='sad':
detect_faces('sad')
else:
detect_faces('surprise')
#Concentration Index Calculation
CI=[]
for i in range(len(mode_emotion_probabilities)):
if mode_emotion_indices[i] == 0:
ci = (mode_emotion_probabilities[i] * 0.5) * 100
CI.append(ci)
elif mode_emotion_indices[i] == 1:
ci = ( mode_emotion_probabilities[i] * 0) * 100
CI.append(ci)
elif mode_emotion_indices[i] == 2:
ci = ( mode_emotion_probabilities[i] * 0.12) * 100
CI.append(ci)
elif mode_emotion_indices[i]== 3:
ci = (mode_emotion_probabilities[i] * 0.68) * 100
CI.append(ci)
elif mode_emotion_indices[i]== 4:
ci = ( mode_emotion_probabilities[i] * 0.75) * 100
CI.append(ci)
elif mode_emotion_indices[i]== 5:
ci = ( mode_emotion_probabilities[i] * 0.628) * 100
CI.append(ci)
else:
ci = ( mode_emotion_probabilities[i] * 0.628) * 100
CI.append(ci)
62
#Engagement Classification
if avg_CI>50:
print("HIGHLY ENGAGED...!!!")
classification(avg_CI)
elif avg_CI>1 and avg_CI<=50:
classification(avg_CI)
print("NOMINALLY ENGAGED...!!!")
else:
classification(avg_CI)
print("NOT ENGAGED...!!!")