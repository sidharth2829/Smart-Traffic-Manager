# Import necessary packages
import numpy as np
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
from scipy import spatial
import cv2
from input_retrieval import *
import multiprocessing
from ultralytics import YOLO
from atcs import traffic_control
# All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck"]
FRAMES_BEFORE_CURRENT = 5  # Frames to remember for tracking
inputWidth, inputHeight = 640, 640  # YOLOv11 default size

#Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold, USE_GPU, inputVideoPathList, outputVideoPathAll= parseCommandLineArguments()


# Load YOLOv11 model
print("[INFO] Loading YOLOv11 model...")
model = YOLO('yolo11n.pt')

# Display vehicle count on the frame
def displayVehicleCount(frame, vehicle_count, lane):
    cv2.putText(
        frame,
        'Detected Vehicles in Lane' + str(lane+1) + " = " + str(vehicle_count),
        (20, 20 + lane * 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

# Draw detection boxes
def drawDetectionBoxes(boxes, classIDs, confidences, frame):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        color = (0, 255, 0)  # Green for all vehicles
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Convert numpy values to Python native types
        label = LABELS[classIDs[i]]
        confidence = float(confidences[i])  # Ensure it's a float
        text = f"{label}: {confidence:.2f}"

        # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Count vehicles crossing the line
def count_vehicles(boxes, classIDs, vehicle_count, previous_frame_detections):
    current_detections = {}
    for i, box in enumerate(boxes):
        x, y, w, h = box
        centerX = x + (w // 2)
        centerY = y + (h // 2)

        # Count only vehicles in the list
        if LABELS[classIDs[i]] in list_of_vehicles:
            current_detections[(centerX, centerY)] = vehicle_count
            if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections):
                vehicle_count += 1

    return vehicle_count, current_detections

# Check if box was in previous frames
def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
    centerX, centerY, width, height = current_box
    dist = np.inf

    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:
            continue
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if temp_dist < dist:
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if dist > (max(width, height) / 2):
        return False

    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    return True

class TrackCount:
	_instance = None

	def __init__(self):
		self.vehicle_lane_count = multiprocessing.Manager().list([0,0,0,0])
		print(self.vehicle_lane_count)

	def __new__(self):
		if self._instance is None:
			self._instance = super().__new__(self)
		return self._instance

	def update_count(self,lane,value):
		self.vehicle_lane_count[lane] = value

	def reset_count(self,lane):
		self.vehicle_lane_count[lane] = 0

	def get_count(self,lane):
		return self.vehicle_lane_count[lane]

# Main detection and counting function
def yolo_detection_counter(vehicle_count_instance, lane, inputVideoPath, outputVideoPath):
    videoStream = cv2.VideoCapture(inputVideoPath)
    fps = FPS().start()
    time.sleep(1.0)

    previous_frame_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]
    fvs = FileVideoStream(inputVideoPath).start()
    start_time = int(time.time())

    try:
        while fvs.more():
            frame = fvs.read()
            if frame is None:
                break

            results = model.predict(frame, conf=0.25,verbose=False)
            boxes, classIDs, confidences = [], [], []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf.cpu().numpy()
                    classID = int(box.cls.cpu().numpy())

                    if conf > preDefinedConfidence:
                        boxes.append([x1, y1, x2 - x1, y2 - y1])
                        confidences.append(conf)
                        classIDs.append(classID)

            drawDetectionBoxes(boxes, classIDs, confidences, frame)
            vehicle_count, current_detections = count_vehicles(boxes, classIDs, vehicle_count_instance.get_count(lane), previous_frame_detections)
            displayVehicleCount(frame, vehicle_count, lane)
            vehicle_count_instance.update_count(lane, vehicle_count)

            cv2.imshow('Frame', cv2.resize(frame, (1200, 800)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            fps.update()
            previous_frame_detections.pop(0)
            previous_frame_detections.append(current_detections)

    except Exception as error:
        print("[ERROR]", error)
    finally:
        fps.stop()
        print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        fvs.stop()
        videoStream.release()
        return

if __name__ == '__main__':
	vehicle_count_instance = TrackCount()
	
	process1 = multiprocessing.Process(target=yolo_detection_counter, args=(vehicle_count_instance,0,inputVideoPathList[0],outputVideoPathAll[0]))
	process3 = multiprocessing.Process(target=yolo_detection_counter, args=(vehicle_count_instance,1,inputVideoPathList[1],outputVideoPathAll[1]))
	process4 = multiprocessing.Process(target=yolo_detection_counter, args=(vehicle_count_instance,2,inputVideoPathList[2],outputVideoPathAll[2]))
	process5 = multiprocessing.Process(target=yolo_detection_counter, args=(vehicle_count_instance,3,inputVideoPathList[3],outputVideoPathAll[3]))
	process2 = multiprocessing.Process(target=traffic_control, args=(vehicle_count_instance,))
	
	process1.start()
	process3.start()
	process4.start()
	process5.start()
	process2.start()

	process1.join()
	process3.join()
	process4.join()
	process5.join()
	process2.join()
