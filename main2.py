import numpy as np
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
from scipy import spatial
import cv2
from input_retrieval import *
import multiprocessing

list_of_vehicles = ["bicycle","car","motorbike","bus","truck"]
FRAMES_BEFORE_CURRENT = 5
inputWidth, inputHeight = 416, 416

LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold, USE_GPU, _, _ = parseCommandLineArguments()

inputVideoPathList = [inputVideoPath]
outputVideoPathAll = [outputVideoPath]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

def displayVehicleCount(frame, vehicle_count, lane):
	cv2.putText(frame, 'Detected Vehicles in Lane' + str(lane+1) + "::  =  " + str(vehicle_count), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.FONT_HERSHEY_COMPLEX_SMALL)

def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
	x1_line, y1_line, x2_line, y2_line = line_coordinates
	if (x_mid_point >= x1_line and x_mid_point <= x2_line+5) and (y_mid_point >= y1_line and y_mid_point <= y2_line+5):
		return True
	return False

def displayFPS(start_time, num_frames):
	current_time = int(time.time())
	if(current_time > start_time):
		num_frames = 0
		start_time = current_time
	return start_time, num_frames

def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 255, 0), thickness=2)

def initializeVideoWriter(video_width, video_height, videoStream, outputVideoPath):
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps, (video_width, video_height), True)

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0:
			continue
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]
	if (dist > (max(width, height)/2)):
		return False
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
	current_detections = {}
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			centerX = x + (w//2)
			centerY = y+ (h//2)
			if (LABELS[classIDs[i]] in list_of_vehicles):
				current_detections[(centerX, centerY)] = vehicle_count
				if not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections):
					vehicle_count += 1
				ID = current_detections.get((centerX, centerY))
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1
				cv2.putText(frame, str(ID), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
	return vehicle_count, current_detections

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

class TrackCount:
	_instance = None
	def __init__(self):
		self.vehicle_lane_count = multiprocessing.Manager().list([0,0,0,0])
	def __new__(self):
		if self._instance is None:
			self._instance = super().__new__(self)
		return self._instance
	def update_count(self, lane, value):
		self.vehicle_lane_count[lane] = value
	def reset_count(self, lane):
		self.vehicle_lane_count[lane] = 0
	def get_count(self, lane):
		return self.vehicle_lane_count[lane]

def yolo_detection_counter(vehicle_count_instance, lane, inputVideoPath, outputVideoPath):
	videoStream = cv2.VideoCapture(inputVideoPath)
	fps = FPS().start()
	time.sleep(1.0)
	video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
	video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	x1_line = 0
	y1_line = video_height//2
	x2_line = video_width
	y2_line = video_height//2
	previous_frame_detections = [{(0,0):0} for _ in range(FRAMES_BEFORE_CURRENT)]
	fvs = FileVideoStream(inputVideoPath).start()
	writer = initializeVideoWriter(video_width, video_height, videoStream, outputVideoPath)
	start_time = int(time.time())
	num_frames = 0
