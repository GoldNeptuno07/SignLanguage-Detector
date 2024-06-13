import csv
import mediapipe as mp
import cv2
import numpy as np


def main(path, target, samples):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_style = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands= 1)

    # Vector to save the coords
    landmarks_vect = []
    counter = 0

    while True:
        counter += 1
        data, image = cap.read()
        # Flip the Image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # Store the Results
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 255, 0), thickness=10, circle_radius=10),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=15)
                )

                # Landmarks
                row = []
                for mark in hand_landmarks.landmark:
                    x = mark.x
                    y = mark.y
                    z = mark.z
                    row.append([x,y,z])
                row = np.array(row)
                distances = np.sqrt(np.sum(np.power(row[:, np.newaxis] - row, 2), axis= 2)).reshape(-1,).tolist()
                distances.append(target)
                landmarks_vect.append(distances)

        cv2.putText(image, f'{str(counter)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, thickness= 5, fontScale= 1, color= (200,255,200))
        cv2.imshow('Sign Language', image)

        if (cv2.waitKey(1) & 0xFF == ord('q')) | counter == samples:
            break

    # Save the Data
    with open(f'CSV/{path}', 'w') as file:
        writer = csv.writer(file)
        for coord in landmarks_vect:
            writer.writerow(coord)


"""
    If you're planning to train the model from scratch, you must 
    create the data for each sign running this script, and make sure to 
    modify the path and target for each sign (A, B, Open Hand, etc).
"""
if __name__ == "__main__":
    main(path= 'A.csv', target= 'A', samples= 250)
