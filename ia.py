from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

stream = CamGear(source='https://youtube.com/shorts/jq2-m29tRW4?si=oJxg3KMi1NHn2bO2', stream_mode=True, logging=True).start()

while True:
    frame = stream.read()

    if frame is None:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
stream.stop()
