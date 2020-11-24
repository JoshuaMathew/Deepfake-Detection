# -*- coding: utf-8 -*-
"""liteflow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1se-1HOp_N0GbGR6HaELcsmmqsgJ11Zrg
"""

#create liteflow model for optical flow generation

import torch

import getopt
import math
import numpy
import os
os.chdir('/content/drive/My Drive/MAE 496/liteflow/pytorch')
import PIL
import PIL.Image
import sys
from correlation import correlation

model_path = '/content/drive/My Drive/MAE 496/liteflow/pytorch/network-default.pytorch'

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.size()) not in backwarp_tenGrid:
		tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
		tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Features(torch.nn.Module):
			def __init__(self):
				super(Features, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tenInput):
				tenOne = self.netOne(tenInput)
				tenTwo = self.netTwo(tenOne)
				tenThr = self.netThr(tenTwo)
				tenFou = self.netFou(tenThr)
				tenFiv = self.netFiv(tenFou)
				tenSix = self.netSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
			# end
		# end

		class Matching(torch.nn.Module):
			def __init__(self, intLevel):
				super(Matching, self).__init__()

				self.fltBackwarp = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

				if intLevel != 2:
					self.netFeat = torch.nn.Sequential()

				elif intLevel == 2:
					self.netFeat = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
						torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
					)

				# end

				if intLevel == 6:
					self.netUpflow = None

				elif intLevel != 6:
					self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)

				# end

				if intLevel >= 4:
					self.netUpcorr = None

				elif intLevel < 4:
					self.netUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)

				# end

				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
				)
			# end

			def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
				tenFeaturesFirst = self.netFeat(tenFeaturesFirst)
				tenFeaturesSecond = self.netFeat(tenFeaturesSecond)

				if tenFlow is not None:
					tenFlow = self.netUpflow(tenFlow)
				# end

				if tenFlow is not None:
					tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, tenFlow=tenFlow * self.fltBackwarp)
				# end

				if self.netUpcorr is None:
					tenCorrelation = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFeaturesFirst, tenSecond=tenFeaturesSecond, intStride=1), negative_slope=0.1, inplace=False)

				elif self.netUpcorr is not None:
					tenCorrelation = self.netUpcorr(torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFeaturesFirst, tenSecond=tenFeaturesSecond, intStride=2), negative_slope=0.1, inplace=False))

				# end

				return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation)
			# end
		# end

		class Subpixel(torch.nn.Module):
			def __init__(self, intLevel):
				super(Subpixel, self).__init__()

				self.fltBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

				if intLevel != 2:
					self.netFeat = torch.nn.Sequential()

				elif intLevel == 2:
					self.netFeat = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
						torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
					)

				# end

				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=[ 0, 0, 130, 130, 194, 258, 386 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
				)
			# end

			def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
				tenFeaturesFirst = self.netFeat(tenFeaturesFirst)
				tenFeaturesSecond = self.netFeat(tenFeaturesSecond)

				if tenFlow is not None:
					tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, tenFlow=tenFlow * self.fltBackward)
				# end

				return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([ tenFeaturesFirst, tenFeaturesSecond, tenFlow ], 1))
			# end
		# end

		class Regularization(torch.nn.Module):
			def __init__(self, intLevel):
				super(Regularization, self).__init__()

				self.fltBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

				self.intUnfold = [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]

				if intLevel >= 5:
					self.netFeat = torch.nn.Sequential()

				elif intLevel < 5:
					self.netFeat = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
						torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
					)

				# end

				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				if intLevel >= 5:
					self.netDist = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
					)

				elif intLevel < 5:
					self.netDist = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=([ 0, 0, 7, 5, 5, 3, 3 ][intLevel], 1), stride=1, padding=([ 0, 0, 3, 2, 2, 1, 1 ][intLevel], 0)),
						torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=(1, [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]), stride=1, padding=(0, [ 0, 0, 3, 2, 2, 1, 1 ][intLevel]))
					)

				# end

				self.netScaleX = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
				self.netScaleY = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
			# eny

			def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow):
				tenDifference = (tenFirst - backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackward)).pow(2.0).sum(1, True).sqrt().detach()

				tenDist = self.netDist(self.netMain(torch.cat([ tenDifference, tenFlow - tenFlow.view(tenFlow.shape[0], 2, -1).mean(2, True).view(tenFlow.shape[0], 2, 1, 1), self.netFeat(tenFeaturesFirst) ], 1)))
				tenDist = tenDist.pow(2.0).neg()
				tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

				tenDivisor = tenDist.sum(1, True).reciprocal()

				tenScaleX = self.netScaleX(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
				tenScaleY = self.netScaleY(tenDist * torch.nn.functional.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor

				return torch.cat([ tenScaleX, tenScaleY ], 1)
			# end
		# end

		self.netFeatures = Features()
		self.netMatching = torch.nn.ModuleList([ Matching(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
		self.netSubpixel = torch.nn.ModuleList([ Subpixel(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
		self.netRegularization = torch.nn.ModuleList([ Regularization(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(model_path).items() })
	# end

	def forward(self, tenFirst, tenSecond):
		tenFirst[:, 0, :, :] = tenFirst[:, 0, :, :] - 0.411618
		tenFirst[:, 1, :, :] = tenFirst[:, 1, :, :] - 0.434631
		tenFirst[:, 2, :, :] = tenFirst[:, 2, :, :] - 0.454253

		tenSecond[:, 0, :, :] = tenSecond[:, 0, :, :] - 0.410782
		tenSecond[:, 1, :, :] = tenSecond[:, 1, :, :] - 0.433645
		tenSecond[:, 2, :, :] = tenSecond[:, 2, :, :] - 0.452793

		tenFeaturesFirst = self.netFeatures(tenFirst)
		tenFeaturesSecond = self.netFeatures(tenSecond)

		tenFirst = [ tenFirst ]
		tenSecond = [ tenSecond ]

		for intLevel in [ 1, 2, 3, 4, 5 ]:
			tenFirst.append(torch.nn.functional.interpolate(input=tenFirst[-1], size=(tenFeaturesFirst[intLevel].shape[2], tenFeaturesFirst[intLevel].shape[3]), mode='bilinear', align_corners=False))
			tenSecond.append(torch.nn.functional.interpolate(input=tenSecond[-1], size=(tenFeaturesSecond[intLevel].shape[2], tenFeaturesSecond[intLevel].shape[3]), mode='bilinear', align_corners=False))
		# end

		tenFlow = None

		for intLevel in [ -1, -2, -3, -4, -5 ]:
			tenFlow = self.netMatching[intLevel](tenFirst[intLevel], tenSecond[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
			tenFlow = self.netSubpixel[intLevel](tenFirst[intLevel], tenSecond[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
			tenFlow = self.netRegularization[intLevel](tenFirst[intLevel], tenSecond[intLevel], tenFeaturesFirst[intLevel], tenFeaturesSecond[intLevel], tenFlow)
		# end

		return tenFlow * 20.0
	# end
# end

netNetwork = None

##########################################################

def estimate(tenFirst, tenSecond):
	global netNetwork

	if netNetwork is None:
		netNetwork = Network().cuda().eval()
	# end

	assert(tenFirst.shape[1] == tenSecond.shape[1])
	assert(tenFirst.shape[2] == tenSecond.shape[2])

	intWidth = tenFirst.shape[2]
	intHeight = tenFirst.shape[1]

	#assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
	tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow[0, :, :, :].cpu()

#get optical flow using image input

import numpy as np

def get_flow(im1,im2):
  tenFirst = torch.FloatTensor(numpy.array(im1)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
  tenSecond = torch.FloatTensor(numpy.array(im2)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
	
  tenOutput = estimate(tenFirst, tenSecond)
  tenOutput = np.array(tenOutput)
  tenOutput = np.moveaxis(tenOutput,0,2)
  return tenOutput

#get optical field over face 
import cv2 as cv
import numpy as np

#initialize dnn model for face detection
modelFile = "/content/drive/My Drive/MAE 496/Data/face detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "/content/drive/My Drive/MAE 496/Data/face detection/deploy.prototxt.txt"
net = cv.dnn.readNetFromCaffe(configFile, modelFile)
frameWidth = 300
frameHeight = 300

def get_optical_flow(file, frame_limit, matrix_shape):

    cap = cv.VideoCapture(file)

    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, prev_frame = cap.read()
      
    #resize frames to 300x300 to input into face detector
    prev_frame = cv.resize(prev_frame,(frameWidth,frameHeight))
    
    frame_count = 0

    flows= np.zeros((frame_limit,matrix_shape,matrix_shape,2))

    while(frame_count<frame_limit):
        
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        
        #resize frame to be 300x300
        frame = cv.resize(frame,(frameWidth,frameHeight))
        blob = cv.dnn.blobFromImage(frame, 1.0, (frameWidth, frameHeight), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
             
        #get coordinates of the face
        for i in range(detections.shape[2]):

          confidence = detections[0, 0, i, 2]
          
          if confidence > .3:
            

            startX = int(detections[0, 0, i, 3] * frameWidth * .95)
            startY = int(detections[0, 0, i, 4] * frameHeight * .95)
            endX = int(detections[0, 0, i, 5] * frameWidth * 1.05)
            endY = int(detections[0, 0, i, 6] * frameHeight * 1.05)
        
        #if no face is detected use a default section from the middle of the image
        #if face_count == 0:
        #  print('no face detected')
        #  startX =90
        #  endX = 210
        #  startY = 30
        #  endY = 280

        # generate dense optical flow between frames
        flow = get_flow(prev_frame, frame)
        
        #extract optical flow over the facial region
        try:
          flow_face = flow[startY:endY,startX:endX]  
        except:
          print('low confidence')
          print(frame_count) 
          flow_face = flow[30:280,90:210]  
        
        flow_face = cv.resize(flow_face,(matrix_shape,matrix_shape))

        #store flow field
        flows[frame_count] = flow_face 
        
        # Updates previous frame
        prev_frame = frame
        frame_count +=1

        # Program stops calculating optical flow when frame limit is reached
        if frame_count == frame_limit:
            break
    
    # The following frees up resources
    cap.release()
    
    flows = np.array(flows)
    return flows

def count_files(file_path):
  import os
  catalogs = os.walk(file_path)
  count = 0
  for root, _, files in catalogs:
      for name in files:
        count +=1
  return count

#count  = count_files('/content/drive/My Drive/MAE 496/Data/CelebDF-augmented2/celeb2_videos/o')
#print(count)

#function to generate optical flows for every video in a folder (improved)
#inputs: 
#file_path - path to folder containing o and s folders of videos
#save_path - path to folder where optical flows will be saved
#frame_limit - frames to extract from each video
#matrix_shape - shape of output optical flow field
#output:

import os

def get_data(file_path,save_path,frame_limit,matrix_shape):

    file_count = count_files(file_path)
    catalogs = os.walk(file_path)
    labels = np.zeros(file_count)
    count = 0  

    for root, _, files in catalogs:
        #  get the original videos
        if root[-1] == 'o':
            for name in files:
                vid_path = os.path.join(root, name)
                print(vid_path)
                txt = 'flow-o-{num}'
                file_name = save_path +'/' + txt.format(num = count)
              
                flows = get_optical_flow(vid_path,frame_limit,matrix_shape)
                np.save(file_name,flows)

                labels[count] = 0
                count+=1
                print(count)
              
        #  get the swapped videos
        if root[-1] == 's':
            for name in files:
                vid_path = os.path.join(root, name)
                txt = 'flow-s-{num}'
                file_name = save_path +'/' + txt.format(num = count)
                
                flows = get_optical_flow(vid_path,frame_limit,matrix_shape)
                np.save(file_name,flows)
                
                labels[count] = 1
                count+=1
                print(count)

    return labels

file_path = '/content/drive/My Drive/MAE 496/Data/CelebDF-augmented1/celeb1_videos'
save_path = '/content/drive/My Drive/MAE 496/Data/CelebDF-augmented1/celeb_set1_flows'
frame_limit = 50
matrix_shape = 100 

labels = get_data(file_path,save_path,frame_limit,matrix_shape)
np.save('/content/drive/My Drive/MAE 496/Data/CelebDF-augmented1/labels1',labels)

count = 0
for i in list(range(0,10)):
  path = 'C/yuh/hey'
  txt = 'blah-o-{count}'
  L = path + '/' + txt.format(count = i)
  print(L)

L.rsplit('-')[1]