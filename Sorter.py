import os
import shutil
from ModelWrapper import *
from PIL import Image
import time


class Sorter:
	""" Class that sorts a folder of images given a
	model to do the inference """


	def __init__(self,modellocation,rawImageFolder, sortedFold="SortedData",classAliases=None):
		self.wrap = ModelWrapper(modellocation,classAliases)
		self.rawPath = rawImageFolder
		self.sortedFold=sortedFold
		if(not os.path.exists(self.sortedFold)):
			os.mkdir(self.sortedFold)
		if(classAliases is None):
			self.classAliases=range(self.wrap.get_nbClass)
		elif(len(classAliases)!=self.wrap.get_nbClass()):
			print("incompatible classAliases list, reverting to default")
			self.classAliases=range(self.wrap.get_nbClass)
		else:
			self.classAliases=classAliases

		if(set(os.listdir(sortedFold))!= set(classAliases)):
			self.reset_sorted()

	def reset_sorted(self):
		for fold in os.listdir(self.sortedFold):
			for im in os.listdir(os.path.join(self.sortedFold,fold)):
				shutil.move(os.path.join(self.sortedFold,fold,im),self.rawPath)
			os.rmdir(os.path.join(self.sortedFold,fold))

		for alia in self.classAliases:
			os.mkdir(os.path.join(self.sortedFold,alia))

	def sortImg(self,imgPath):
		try:
			with Image.open(imgPath) as im:
				im = im.convert('RGB')
				result = self.wrap.predict(im)
				shutil.move(imgPath,os.path.join(self.sortedFold,result))
		except Exception as ex:
			print("more info : ",dir(ex))
			print("skipping this file")
			time.sleep(70)

	def sortAll(self,folder=None):
		if(folder is None):
			folder=self.rawPath

		for imgPath in os.listdir(folder):
			if(os.path.isfile(os.path.join(folder,imgPath))):
				self.sortImg(os.path.join(folder,imgPath))
