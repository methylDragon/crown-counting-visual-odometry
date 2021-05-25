import time

import cv2
import ccvo

import mos.utils

cap = cv2.VideoCapture(2)
cap.set(38, 1) # Set capture buffer size to 0

FPS = 60
M_CROWN_DISTANCE = 0.03

tracker = ccvo.CCVO({'static_centroid_distance_estimation': True})
last_time = time.perf_counter()

while True:
    ret, image = cap.read()

    if ret:
        tracker.update(image)

    current_time = time.perf_counter()
    time_delta, last_time = current_time - last_time, current_time

    for name, img in tracker.visualisation_images.items():
        cv2.imshow(name, img)

    info = tracker.get_info()

    # Compute velocity =========================================================
    estimated_centroid_distance = info['estimated_centroid_distance']
    m_per_px = M_CROWN_DISTANCE / estimated_centroid_distance
    m_vel_per_sec = info['average_pixel_vel'] / time_delta * m_per_px

    print(m_vel_per_sec, end="                               \r", flush=True)

    key = cv2.waitKey(
        mos.utils.clamp(int(1000 // FPS - time_delta * 1000), smallest=1)
    )

    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
