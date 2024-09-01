import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=100)
mp_drawing = mp.solutions.drawing_utils

# Function to resize the frame while maintaining aspect ratio
def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

# Function to process hand detection
def process_hand_detection(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    any_save_signal = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tips_indices = [4, 8, 12, 16, 20]
            finger_states = []
            for index in tips_indices:
                tip_landmark = hand_landmarks.landmark[index]
                knuckle_landmark = hand_landmarks.landmark[index - 2]

                if index == 4:
                    is_open = tip_landmark.x > hand_landmarks.landmark[8].x
                else:
                    is_open = tip_landmark.y < knuckle_landmark.y

                finger_states.append(1 if is_open else 0)

            if finger_states == [0, 1, 1, 0, 1]:
                any_save_signal = True

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return any_save_signal, image

# Function to process background_change
def detect_background_change(reference_image_path, current_frame):
    # Read the reference image
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        raise ValueError(f"Reference image not found at '{reference_image_path}'.")

    # Convert frames to grayscale
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors for both frames
    keypoints1, descriptors1 = orb.detectAndCompute(current_frame_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(reference_gray, None)

    # Check if descriptors are valid
    if descriptors1 is None or descriptors2 is None:
        return 0.0  # Handle case where keypoint detection failed

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Get matched keypoints
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    H, _ = cv2.findHomography(matched_keypoints1, matched_keypoints2, cv2.RANSAC, 5.0)

    # Apply homography to align the current frame with the reference frame
    aligned_img = cv2.warpPerspective(current_frame, H, (current_frame.shape[1], current_frame.shape[0]))

    # Crop borders from aligned and reference images
    border_percent = 0.10
    h, w = reference_image.shape[:2]
    border_h = int(border_percent * h)
    border_w = int(border_percent * w)

    aligned_img_cropped = aligned_img[border_h:h-border_h, border_w:w-border_w]
    reference_img_cropped = reference_image[border_h:h-border_h, border_w:w-border_w]

    # Calculate absolute difference between aligned and reference images
    diff_img = cv2.absdiff(reference_img_cropped, aligned_img_cropped)
    diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)

    # Calculate threshold and thresholded difference image
    threshold = np.mean(diff_gray) * 14  # Adjust threshold as needed
    _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    # Calculate percentage change based on thresholded image
    total_pixels = np.sum(diff_thresh > 0)  # Count non-zero pixels after thresholding
    change_percentage = (total_pixels / (reference_image.shape[0] * reference_image.shape[1])) * 100

    return change_percentage
#for drawing boxes in entry and exit
def draw_boxes(img, boxes, confidences, class_ids):
    for (box, conf, class_id) in zip(boxes, confidences, class_ids):
        if class_id == 0:  # Only draw boxes for persons
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Person: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def entryandexit(frame):
    results = model1(frame)[0]
    
    # Extract bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = result
        if int(class_id) == 0:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            confidences.append(float(conf))
            class_ids.append(int(class_id))
    
    # Draw bounding boxes and count persons
    person_count = len(boxes)
    draw_boxes(frame, boxes, confidences, class_ids)
    
    # Display the frame with bounding boxes
    cv2.putText(frame, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return person_count


# Function to find the median
def findmedian(counts):
    return np.median(counts)

# Function to count people in a frame and manage the people count array
def people_count_of_frame(image, model, people_count_array):
    confidence_threshold = 0
    NMS_THRESHOLD = 0.4
    classes, scores, boxes = model.detect(image, confidence_threshold, NMS_THRESHOLD)
    count = len(classes)
    people_count_array.append(count)
    if len(people_count_array) == 5:
        people_count_array.clear()  # Clear all elements from the array
    return people_count_array

# Create Tkinter window
root = tk.Tk()
root.title("hand and crowd detection")
root.geometry("1400x1600")


# Create labels to display the images
label_positions = [
    (45, 0), (501, 0), (957, 0), (501, 292)
]  # Positions for the video frames
image_labels = []
for i, position in enumerate(label_positions):
    label = tk.Label(root)
    label.place(x=position[0], y=position[1], width=432, height=243)  # Position and dimensions of each label
    image_labels.append(label)
# Create labels to display the crowd counts
crowd_count_label = tk.Label(root, text="Crowd Count (Video 1): ", bg="#000000", fg="white")

# Position the crowd count labels
crowd_count_label.place(x=210, y=250)   # Position for Video   

# Load and resize the caution image
caution_img = Image.open("caution symbol.jpg")
caution_img = caution_img.resize((50, 50), Image.LANCZOS)  # Resize caution image
caution_tk = ImageTk.PhotoImage(caution_img)

# Create labels to display the caution images
caution_labels = [tk.Label(root, image=caution_tk, bg="#000000") for _ in range(4)]

# Create labels to display the background change percentages
bg_change_label = tk.Label(root, text="Background Change (Video 2): ", bg="#000000", fg="white")

# Position the background change percentage labels
bg_change_label.place(x=600, y=250)   # Position for Video 2

# Create labels to display the people count for the 7th and 8th videos
people_count_label = tk.Label(root, text="People Count (Video 3): ", bg="#000000", fg="white")

# Position the people count labels for the 7th and 8th videos
people_count_label.place(x=1100, y=250)   # Position for Video 3
#create labels to display count for webcam
live_webcam_label = tk.Label(root, text="People count (Video 4): ", bg="#000000", fg="white")
#position for live webcam label for fourth video
live_webcam_label.place(x=600, y=540) #position for video 4




# Paths to the video files
video_paths = ["crowd2.mp4", "output_video.mp4","crowd2.mp4", 0]

# Create instances of VideoCapture
caps = [cv2.VideoCapture(path) for path in video_paths]

# Crowd counting model setup
cfgpath = r'/Users/suhanigoli/Documents/pythonprojects/model.cfg'
weightspath = r'/Users/suhanigoli/Documents/pythonprojects/model1.weights'
net = cv2.dnn.readNet(cfgpath, weightspath)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)
model1 = YOLO('yolov8n.pt')
entryexitcount= 0
frame_count = [0, 0, 0,0]
skip = [0, 0, 0, 0]
people_count_arrays = [ ]
crowd_threshold = 70  # Set your threshold value here

# Paths to store reference images
reference_image_paths = ["reference_image_{}.jpg".format(i) for i in range(1)]

# Flags to check if reference images have been captured
reference_images_captured = [False] 

while True:
    all_ret = True  # Flag to check if all videos are still running
    
    for cap_index, cap in enumerate(caps):
        ret, image = cap.read()
        if not ret:
            # If a video stops, continue without breaking the loop
            all_ret = False
            continue

        # Resize the frame
        image_resized = resize_frame(image, 432, 243)

        # Perform hand detection and crowd counting for the first three videos
        if cap_index == 0:
            any_save_signal, image_resized = process_hand_detection(image_resized)
            if any_save_signal:
                caution_labels[cap_index].place(x=label_positions[cap_index][0] + 191, y=label_positions[cap_index][1] + 250, width=50, height=50)
            else:
                caution_labels[cap_index].place_forget()
            frame_count[cap_index] += 1
            if frame_count[cap_index] % 10 == 0:
                robust_median = findmedian(people_count_arrays)
                people_count_arrays = []
                if robust_median > crowd_threshold:
                        caution_labels[cap_index].place(x=label_positions[cap_index][0] + 191, y=label_positions[cap_index][1] + 250, width=50, height=50)
                else:
                    caution_labels[cap_index].place_forget()

                crowd_count_label.config(text="Crowd Count (Video {}): {:.2f}".format(cap_index + 1, robust_median))
            else:
                people_count_arrays = people_count_of_frame(image, model, people_count_arrays)

        # Perform background subtraction for the next three videos (4, 5, 6)
        elif cap_index==1:
            frame_count[cap_index]+=1
            # Capture the first frame as the reference image
            if not reference_images_captured[cap_index - 1]:
                cv2.imwrite(reference_image_paths[cap_index - 1], image_resized)
                reference_images_captured[cap_index - 1] = True
            # Detect background change
            if frame_count[cap_index ] % 10== 0:
                change_percentage = detect_background_change(reference_image_paths[cap_index - 1], image_resized)
                bg_change_label.config(text="Background Change (Video {}): {:.2f}%".format(cap_index + 1, change_percentage))  # Update background change label

                if change_percentage > 0.3:
                    caution_labels[cap_index].place(x=label_positions[cap_index][0] + 191, y=label_positions[cap_index][1] + 250, width=50, height=50)
                else:
                    caution_labels[cap_index].place_forget()
        elif cap_index==2:

            frame_count[cap_index] += 1

            height,width,channels = image.shape
            mask_top = int(width*0.2)
            image[:,:mask_top]=0
            image[:,-mask_top:]=0
            if frame_count[cap_index]%30==0:
                count = entryandexit(image)
                entryexitcount+=count
            people_count_label.config(text="People Count (Video {}): {}".format(cap_index + 1, entryexitcount))

        elif cap_index ==3:
            any_save_signal, image_resized = process_hand_detection(image_resized)
            if any_save_signal:
                caution_labels[cap_index].place(x=label_positions[cap_index][0] + 191, y=label_positions[cap_index][1] + 250, width=50, height=50)
            else:
                caution_labels[cap_index].place_forget()
            count1 = entryandexit(image)  
            live_webcam_label.config(text="crowd count (video{}): {}".format(cap_index+1, count1))      

            

            
        # Convert the OpenCV image to PIL format
        img_pil = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

        # Convert the PIL image to Tkinter format
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Update the image label with the new image
        image_labels[cap_index].config(image=img_tk)
        image_labels[cap_index].image = img_tk

    # Update the Tkinter window
    root.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop on 'q' key press
        break    
# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
root.mainloop()
