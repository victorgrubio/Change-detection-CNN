import os
import argparse
from cv2 import destroyAllWindows
from sys import exit
from datetime import datetime
from compiledSegmentationPredict import predictImage,predictFolder
from modelPreprocessing import loadModel

if __name__ == '__main__':

	#Input arguments and options
	parser = argparse.ArgumentParser()
	parser.add_argument('model_path',help='Path to model')
	parser.add_argument('--mode',type=str,required=True,help='specify video,image_folder or image mode')
	parser.add_argument('--debug',action='store_true',help='debug mode - to be determined')
	parser.add_argument('--align',type=bool,help='applies alignment to images',default=True)
	parser.add_argument('--no_display',help='images are not shown during prediction',action='store_true')
	parser.add_argument('--save_results','-s',help='save video or image result of prediction',action='store_true')
	args = parser.parse_args()

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	model = loadModel(args.model_path)
	date = "{:%d_%m-%H_%M}".format(datetime.now())

	try:
		if args.mode == 'image':
			predictImage(model,args)
			
		elif args.mode == 'image_folder':
			predictFolder(model,args)
			
	#CTRL+C handler		
	except KeyboardInterrupt:
		#Exit video recording after analysis has been completed 
		print('User has interrupted the program ...')
		if args.save_results:
		 	#As video_name is date.avi, we can easily remove it using:
			os.remove('.'.join([date,'avi']))
			print('Video file has been deleted ... ')
		destroyAllWindows()
		exit(1)
	
	exit(0)