import cv2
import numpy
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")
col1, col2 = st.columns([2,1])
with col1:
    run = st.checkbox('Run', value = True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text_area = st.title('Answer')
    output_text_area = st.subheader("")

genai.configure(api_key="AIzaSyDc_k8MQk5bE4P-Dk1Ew7NkavvJJhGGBK4")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.9, minTrackCon=0.7)

def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255,0,255), 10)
    elif fingers == [1,0,0,0,0]:
        canvas = numpy.zeros_like(img)

    return current_pos, canvas

def sentToAI(model, canvas, fingers):
    if fingers == [1,1,1,1,0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text

prev_pos = None
canvas = None
ans_txt=""

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = numpy.zeros_like(img)

    # get hand info
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        ans_txt = sentToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if ans_txt:
        output_text_area.text(ans_txt)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)
