import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Load the Model
network = load_model('SignLanguage.keras')

def get_distances(X: list):
    """
        Compute the L2 Norm of the distances between
        each landmark on your hand.
    """
    ind_remove = [
        0,  22,  44,  66,  88, 110, 132, 154, 176, 198, 220, 242, 264,
       286, 308, 330, 352, 374, 396, 418, 440]

    X_array = np.array(X, dtype= np.float32)
    distances = np.sqrt(np.sum(np.power(X_array[:, np.newaxis] - X_array, 2), axis=2)).reshape(1, -1)
    distances = np.delete(distances, ind_remove, axis= 1)
    return distances

def predict(landmarks):
    """
        If you're trying to predict another kind of labels, you must change the labels list
        based on the targets and its corresponding class.
    """
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    Xs = get_distances(landmarks)
    pred_output = np.argmax(network.predict(Xs), axis= 1)[0]
    return labels[pred_output]


def main():
    """
        Function to display camera, predict each video frame and display
        the corresponding class based on the sign.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1)

    row = [[0.,0.,0.]] * 21
    while True:
        data, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmark, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=5, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=10)
                )

                for i, mark in enumerate(hand_landmark.landmark):
                    x = mark.x
                    y = mark.y
                    z = mark.z
                    row[i] = [x, y, z]

        # Display Rectangle
        if results.multi_hand_landmarks:
            x = int(row[9][0] * image.shape[1]) - 200
            y = int(row[9][1] * image.shape[0]) - 200
            sign = predict(row)
            cv2.putText(image, sign, (x, y - 50), color= (0,0,0), thickness= 5, fontScale= 1, fontFace= cv2.FONT_HERSHEY_SIMPLEX)


        cv2.imshow('HandTracker', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
