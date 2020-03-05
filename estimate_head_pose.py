"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import cv2
import numpy as np

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
import math

print("OpenCV version: {}".format(cv2.__version__))

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128

real_width = 114
focal_length = 474

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


def sharpen(gray):
    blurred = gray.copy()
    cv2.GaussianBlur(gray, (5, 5), 0.8, blurred)
    sharpened = gray.copy()
    cv2.addWeighted(gray, 1.5, blurred, 0.5, 0, sharpened)
    return sharpened


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


def adjust_angle(eye_frame_part, x, y):
    adjusted_x = x - eye_frame_part.shape[0] / 2
    adjusted_y = y - eye_frame_part.shape[1] / 2

    proportion_x = adjusted_x / eye_frame_part.shape[0]
    proportion_y = adjusted_y / eye_frame_part.shape[1]

    vertical_max_angle = 75
    horizontal_max_angle = 75

    return proportion_x * horizontal_max_angle, proportion_y * vertical_max_angle


def process_eyes(frame, landmarks):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = sharpen(gray)

    left_eye = gray[int(landmarks[38][1]) - 7: int(landmarks[41][1]) + 7, int(landmarks[36][0]) - 7: int(landmarks[39][0]) + 7]
    right_eye = gray[int(landmarks[44][1]) - 7: int(landmarks[46][1]) + 7, int(landmarks[42][0]) - 7: int(landmarks[45][0]) + 7]

    eyes = [left_eye, right_eye]

    centers = []
    origins = [[int(landmarks[36][0]), int(landmarks[38][1])], [int(landmarks[42][0]), int(landmarks[44][1])]]

    for eye, origin in zip(eyes, origins):
        circles = cv2.HoughCircles(eye, cv2.HOUGH_GRADIENT, 1, eye.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)
        eye_center = (eye.shape[0] / 2, eye.shape[1] / 2)
        if circles is None:
            # no circles were detected on either or both eyes
            # so return prematurely
            return {}
        circles = np.uint16(np.around(circles))
        needed_circles = circles[0, :].tolist()
        needed_circles.sort(key=lambda circle: math.sqrt((eye_center[0] - circle[0])**2 + (eye_center[1] - circle[1])**2))
        the_circle = needed_circles[0]
        # simplePutText(eye, str(len(circles)))
        # for i in circles[0, :]:
        #     cv2.circle(eye, (i[0], i[1]), i[2], (255, 255, 255), 2)
        angles = adjust_angle(eye, the_circle[0], the_circle[1])
        cv2.circle(frame, (the_circle[0] + origin[0], the_circle[1] + origin[1]), 2, (255, 255, 255))
        centers.append(angles)

    centers = {'right': centers[0], 'left': centers[1]}
    return centers


def angle_to_radian(angle_degrees):
    return math.pi * angle_degrees / 180


def cvt_to_radians(eye_angles):
    right = eye_angles['right']
    eye_angles['right'] = (angle_to_radian(right[0]), angle_to_radian(right[1]))
    
    left = eye_angles['left']
    eye_angles['left'] = (angle_to_radian(left[0]), angle_to_radian(left[1]))

    return eye_angles


def main():
    """MAIN"""
    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()

        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks([face_img])
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            mark_detector.draw_marks(
                frame, marks, color=(0, 255, 0))
            right_corner = tuple([int(i) for i in marks[36]])
            left_corner = tuple([int(i) for i in marks[45]])
            # print(marks[36], marks[45])
            cv2.line(frame, right_corner, left_corner, (255, 0, 0), 2)

            pixel_distance = int(math.sqrt((right_corner[0] - left_corner[0]) ** 2 + (right_corner[1] - left_corner[1]) ** 2))
            estimated_distance = (real_width * focal_length) / pixel_distance

            cv2.putText(frame, str(round(estimated_distance, 2)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))


            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            # Uncomment following line to draw pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #    frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

            # Uncomment following line to draw head axes on frame.
            print(steady_pose[1])
            pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])

            angles = process_eyes(frame, marks)
            if bool(angles) is True:
                # print(angles)
                angles = cvt_to_radians(angles)

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            break

    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()


if __name__ == '__main__':
    main()
