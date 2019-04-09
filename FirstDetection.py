from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path,"C:\\Users\\nishant\\Desktop\\courses\\simple-object-tracking\\resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,"C:\\Users\\nishant\\Desktop\\courses\\ROI Example.PNG"), output_image_path=os.path.join(execution_path,"C:\\Users\\nishant\\Desktop\\courses\\imagenew.jpg"))

for eachObject in detections:
	name=eachObject["name"]
	with open('people.csv', 'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow(eachObject["name"] , " : " , eachObject["percentage_probability"],":",eachObject["box_points"])
		
    print(eachObject["name"] , " : " , eachObject["percentage_probability"],":",eachObject["box_points"] )
writeFile.close()