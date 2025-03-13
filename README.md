# Dense-Optical-Flow
Dense Optical Flow in Collab. So that you know, you need to import the dependent libraries before this step and also do not forget to mount the drive that has the data stored.

# 1. Open a video file or capture device, Please substitute your file here
video_path="/content/drive/MyDrive/Colab Notebooks/outputtrim1.mp4"
cap = cv2.VideoCapture(video_path)

# Read the first frame and convert it to grayscale
ret, frame1 = cap.read()
if not ret:
    print("Could not read the first frame!")
    cap.release()
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Prepare an HSV image for display flow
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255  # Saturation set to maximum

while True:
    ret, frame2 = cap.read()
    if not ret:
        break  # End of video

    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 2. Compute Dense Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        prvs, next_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # 3. Convert flow to polar coordinates: magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    # Hue corresponds to flow direction (angle), Value to flow magnitude
    hsv[..., 0] = ang / 2  # Scale angle to fit [0,180] for Hue
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 4. Convert HSV to BGR for display
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Show the color-coded flow
    plt.figure(figsize=(10, 6))
    plt.imshow(bgr)

    # Press Esc to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

    # Update the previous frame
    prvs = next_gray

cap.release()
cv2.destroyAllWindows()
