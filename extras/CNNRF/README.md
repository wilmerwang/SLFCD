## CNN for tumor detection on whole slede images(WSI)  
This project is for the tumor area detection on whole slide images(WSI), using Python 3, keras, and Tensorflow.  
**效果图**  
The repository includes:  

- Preprocess  
- Train and predict   
- Postprocess  
- Random forest  
## Requirements  
- Python 3.6.x or above  
- Tensoflow 1.6.x  
- Openslide  
- sk-learn  
- sk-image  
- open-cv  
- numpy, sciPy  

## Preprocess  
There are have broad categories:  

1. Finding Region of Interest(ROI)  
2. Exatract Patches from ROI  
3. Data augumentation  

**get _ mask.py**  
**get _ patches.py**  

## Keras Finetuming  
Patches classfier for normal and tumor patch  

**train _ images _ classifier.py**  
**evl _ images.py**

## Postprocess  
There are have broad categories:  



1. get consecutive patches for heatmaps  
2. get heatmaps for all trainset slide  

##  Random forest  

This is slide based classifier using what we get in above steps!!   



