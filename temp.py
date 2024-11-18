# import os

# if os.path.exists('models/age_model.h5'):
#     print("File exists!")
# else:
#     print("File not found!")
# import h5py

# try:
#     with h5py.File('models/gender_model.h5', 'r') as f:
#         print("H5 file opened successfully!")
# except Exception as e:
#     print(f"Error opening file: {e}")
# import cv2

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow('Webcam Feed', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
# import cv2

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot access camera")
# else:
#     print("Camera is accessible")
# cap.release()
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
cap.release()


