import tensorflow as tf
import numpy as np
import os
import shutil


class ModelWrapper:
	def __init__(self,modelLocation,classAliases=None,sortedFold="SortedData"):
		""" Wrapper for a CNN discriminating between a set of classes
		classAliases should be a tab of strings, containing the aliases
		of the classes, in order. """
		self.model = tf.keras.models.load_model(modelLocation)
		self.nbClass= self.model.layers[-1].get_config()["units"]
		self.classAliases= range(self.nbClass)


		if(classAliases is not None):
			if(len(classAliases)!=self.nbClass):
				print("Incorrect number of Aliases, ignoring")
			else:
				self.classAliases = classAliases

		
		self.imgsize = self.model.layers[0].get_config()["batch_input_shape"][1:]

	def predict(self,img):
		""" Return prediction as class alias. Input should be a
		single nb.array. Will be resized and rescaled by 1/255
		to fit model's size. """
		rescaled = tf.image.resize(img,self.imgsize[:-1])/255.
		answer = self.model(np.expand_dims(rescaled,axis=0))
		print("FOUND IT, it is : ",self.classAliases[np.argmax(answer)])
		return self.classAliases[np.argmax(answer)]

	def summary(self):
		self.model.summary()

	def get_nbClass(self):
		return self.nbClass

