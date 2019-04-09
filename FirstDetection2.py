from imageai.Detection import VideoObjectDetection
import csv
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
#detector.setModelTypeAsYOLOv3()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path,"C:\\Users\\nishant\\Desktop\\courses\\simple-object-tracking\\resnet50_coco_best_v2.0.1.h5"))
#detector.setModelPath( os.path.join(execution_path , "C:\\Users\\nishant\\Desktop\\courses\\simple-object-tracking\\yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "C:\\Users\\nishant\\Desktop\\courses\\video2.mp4"),
                                output_file_path=os.path.join(execution_path, "C:\\Users\\nishant\\Desktop\\courses\\traffic_mini_detected_3")
                                , frames_per_second=29, log_progress=True)
print(video_path)
for eachObject in detections:
	name=eachObject["name"]
	with open('people.csv', 'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow(eachObject["name"] , " : " , eachObject["percentage_probability"],":",eachObject["box_points"])
		
	print(eachObject["name"] , " : " , eachObject["percentage_probability"],":",eachObject["box_points"])
writeFile.close()