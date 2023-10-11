# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect 
actions = np.array(['Hello', 'Thanks', 'I Love You'])

# data collection for training and testing
# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length 
sequence_length = 30


for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


## Collecting keypoint values for training and testing

cap = cv2.VideoCapture(0) 
# set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #New Loop
    # Loop through actions
    for action in actions:
        #Loop through sequences aka videos
        for sequence in range(no_sequences):
            #Loop thorugh video length aka sequence length
            for frame_num in range(sequence_length):
                
              # Read feed
              ret, frame = cap.read()
              # Make detections  
              image, results = mediapipe_detection(frame, holistic)
              print(results)

              #Draw Landmarks
              draw_styled_landmarks(image, results)

              #New apply wait logic
              if frame_num==0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'COLLECTING FRAMES FOR {} Video Number {}'.format(action, sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
              else:
                cv2.putText(image, 'COLLECTING FRAMES FOR {} Video Number {}'.format(action, sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

              #New export keypoints
              keypoints = extract_keypoints(results)
              npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
              np.save(npy_path, keypoints)

              # break gracefully
              if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows() 

## Data pre-processing and labels and features creation

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window= []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


X = np.array(sequences)
y= to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)



