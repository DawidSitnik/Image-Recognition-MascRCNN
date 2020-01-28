# Image-Recognition-MascRCNN
Project for Image Recognition class. The aim is to detect human on photos and mark them with the border.

## How to Install:

### Using our neural network

1. In PythonAPI directory install dependencies:
```console
	pip3 install -r requirements.txt
 ```
2. Paste photos which you want to validate to the real_test directory
3. Run human_recognition.py script


### Initializing your own neural network:

1. Download datasets and annotations from:
http://cocodataset.org/#download

2. Place annotations to annotation directory 

3. In PythonAPI directory install dependencies:
```console
	pip3 install -r requirements.txt
```

4. Paste photos which you want to validate to the real_test directory

5. Run human_recognition.py script with arguments:
* type of annotations for validation dataset
* type of annotations for test dataset
* directory of training photos directory
* directory of validation photos directory
* directory of photos for real test
* initial weights for network

*So example invocation would look like:*
```console
	python3 new_recognition_script.py ‘val2017’ ‘train2017’ '/media/dawid/MAJKI/train2017'
'/home/dawid/Desktop/human_images'  '../real_test' '../initial_weights/initial_weights.h5'
 ```
