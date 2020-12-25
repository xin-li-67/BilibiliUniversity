# Real World Face and Masked Face Detection  

## Dataset  
All the masked face images and unmasked face images are real world images collected using these three methods:  
1. img_collector.py running on Bing Search API (need Azure access);  
2. Kaggle datasets;  
3. RMFD dataset (https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset);  

## Tech Stack and Frameworks  
- OpenCV
- TensorFlow V2
- Keras
- Caffe-based face detector
- MobileNet V2

## Work  
Training: `python train_mask_detector.py --dataset dataset`  
Image detection: `python detect_mask_image.py --image images/pic.jpeg`  
Video detection: `python detect_mask_video.py`  