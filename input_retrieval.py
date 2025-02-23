import argparse
import os

def parseCommandLineArguments():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input",
		help="path to input video")
	
	ap.add_argument("-o", "--output",
		help="path to output video")
	
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	
	ap.add_argument("-iall", "--inputall",nargs='+',type=str, help="input all 4 files ")

	ap.add_argument("-outputall", "--outputall",nargs='+',type=str, help="outpu all 4 files ")

	ap.add_argument("-t", "--threshold", type=float, default=0.2,
		help="threshold when applying non-maxima suppression")
	
	ap.add_argument("-u", "--use-gpu", type=bool, default=False,
	help="boolean indicating if CUDA GPU should be used")

	args = vars(ap.parse_args())

	# to load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	
	# to derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([args["yolo"], "yolov7-tiny.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov7-tiny.cfg"])
	
	inputVideoPath = args["input"]
	inputVideoPathList = args["inputall"]
	outputVideoPath = args["output"]
	outputVideoPathAll = args["outputall"]
	confidence = args["confidence"]
	threshold = args["threshold"]
	USE_GPU = args["use_gpu"]

	return LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, confidence, threshold, USE_GPU, inputVideoPathList, outputVideoPathAll
