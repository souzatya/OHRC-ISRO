# Automatic-Crater-and-Boulder-Detection-OHRC-ISRO

## Kriti Khare, Soujatya Sarkar

Develop an AI/ML model to detect craters/boulders of all shapes and sizes from the given OHRC data and attach relevant information to the detected craters, preferably in the form of a shape file containing the boundary of the craters/boulders.

Objectives
Design a model which is capable of automatically detecting all craters/boulders in an OHRC irrespective of their shape, size and illumination conditions.

Model is able to quantify the selenographic position and diameter of the detected crater/boulder.

Expected Outcomes
Polygonal shape file or shape-size information of craters/boulders with the location on the given OHRC images.

Dataset Required:
OHRC datasets publicly available in PDS4 format on Chandrayaan Map Browse
Suggested Tools/Technologies:
Python(Programming language), C, C++ or other such programming language, Image processing techniques for crater detection, AI/ML Models for Crater/Boulder Detection.
Expected Solution / Steps to be followed to achieve the objectives:
Selenoreference the downloaded OHRC images using the auxillary information provided with the images with appropriate projection information.
Prepare the training and testing data. And train the model.
Prepare the evaluation metrics for accuracy.
Evaluation Parameters:
Accuracy: Precision of the detected craters/boulders with respect to the actual craters present in the image.
Relevance: Degree to which the detected craters/boulders and associated information match the actual craters.
