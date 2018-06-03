# Tensorflow-object-detection-API-for-face-detcetion
use tensorflow object detection API to detect face.
This project base on tf object detection API and use [wider face dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). 
You can build a object detection project with Tf offical example([object_detection_tutorial.ipynb](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)) step by step.
You can clone this project and training your own dataset or detect custom picture without any other setting.

### 1. clone project
git clone https://github.com/aodiwei/Tensorflow-object-detection-API-for-face-detcetion.git

### 2. install necessary python libary
* tensorflow
* Cython
* opencv

### 3. bulid dataset
* download wider dataset from http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
* data_process/dataset_util.py can help you transfer wider format data to tf record, cause tf API only support tf-record 
* data_process/boundingbox.py can help you understand wider format coordinate to tf record coordinate

### 4. download pre-train model
download from (http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17/.tar.gz) and extract to pre_train_ckp/ssd_mobilenet_v1_coco_11_06_2017

### 5. training
 with default args:
 
```
python3 train.py 
```
 with custom args
 
```
python3 train.py \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt
```    
### 6.eval
```
python3 eval.py
```

### 7. export
export model with 
```export_inference_graph.py```

### 8. detection
You can use my trained model to detect custom image
simply 
```
python3 detection_cv.py --input=your_img.jpg
```
more details please read detection_cv.py

tf offical detection example in detection.py

result sample:

![](/data_process/testpic_box.jpg "test")

### 9. Citation

	@inproceedings{yang2016wider,
	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	Title = {WIDER FACE: A Face Detection Benchmark},
	Year = {2016}}
	
	https://github.com/priya-dwivedi/Deep-Learning/tree/master/tensorflow_toy_detector
	https://github.com/yeephycho/widerface-to-tfrecord
