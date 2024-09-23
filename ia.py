from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO

# aqui o yolo é chamado
model = YOLO('yolov8n.pt')

stream = CamGear(source='https://youtube.com/shorts/jq2-m29tRW4?si=oJxg3KMi1NHn2bO2', stream_mode=True, logging=True).start()

while True:
    frame = stream.read()

    if frame is None:
        break

    # aqui mostra a inferencia do yolo
    results = model(frame)

    # aqui mostra os resultados
    annotated_frame = results[0].plot()

    # aqui mostra o que o yolo detectou
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
stream.stop()
