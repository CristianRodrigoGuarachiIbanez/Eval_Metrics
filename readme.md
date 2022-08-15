

# Evaluation Metrics [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) ![C++](https://img.shields.io/badge/C++-Solutions-red.svg?style=flat&logo=c++) 


This work is a implementation of three different algorithmus in order to evaluate the detection performance on LIDAR-Image of a CNN model.
To evaluate this on test data set, some statistical metrics (Precision, Recall, IoU) were implemented, which are commonly used for the analysis of the object detection.
Here, the output data of the prediction on a sensor dataset  (LIDAR images) ist used, which contains the model predictions as well as the ground truth coordinates 


 Dependencies
------------
 * [Cython]
 * [OpenCV]
 * [Numpy]
 * [Pandas]

## Input data

this implementation uses data from the detection of numbers on LIDAR images. The input data is the result of the implementing YOLO for the detection of a sequence of numbers on the LIDAR images. The YOLO Model is then used to make prediction on a new image data set:

![](02_wb_r_000400_i_yolo3.png)

### Implementation of evaluation metrics

Three different metrics are here implemented to evaluate: 
- whether the probability that an event classified by a model as a sequence of numbers has been correctly classified (Precision).
- The fraction of correctly classified numbers among all the correct classified events and the false alarms (Sensitivity or Recall).

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/d37e557b5bfc8de22afa8aad1c187a357ac81bdb" class="mwe-math-fallback-image-display" aria-hidden="true" style="vertical-align: -5.338ex; width:21.82ex; height:11.843ex;" alt="{\displaystyle {\begin{aligned}{\text{Precision}}&amp;={\frac {tp}{tp+fp}}\\{\text{Recall}}&amp;={\frac {tp}{tp+fn}}\,\end{aligned}}}">

for details check the readme in the file "**documentation**" or click ![here](Yolo_Model_Ev/readme.md)

Lastly, the intersection over union (IoU) was implemented as metric for the evaluation of the overlapping of two matrices. Mainly, it calculates the Jaccard Index and returns a coefficient
of similarity for two sets of coordinates or bounding boxes:

<img alt="" src="https://hasty.ai/media/pages/content-hub/mp-wiki/metrics/iou-intersection-over-union/4c8d99617c-1653640694/6.png" class="mwe-math-fallback-image-display" aria-hidden="true" style="vertical-align: -5.338ex; width:21.82ex; height:11.843ex;" alt="{\displaystyle {\begin{aligned}{\text{Precision}}&amp;={\frac {tp}{tp+fp}}\\{\text{Recall}}&amp;={\frac {tp}{tp+fn}}\,\end{aligned}}}">

### Dataset Generator 
It implements also a image generator to produce synthetic data in form of a test images with printed bounding boxes. 

![](../tests/output_images/image_0_0.png)

Here, the input or reference coordinates for just one bounding box are given. However, a random number of bounding boxes to be printed can be passed as input 
The distance amount the bounding boxes will be randomly generated. 

### Intervention Over Union (IoU)

The IoU is a well known metric, that overlapping area under two different bounding boxes. This measurement based on the Jaccard Index und returns a coefficient
of similarity for two sets of coordinates or bounding boxes:
<img alt="" src="https://hasty.ai/media/pages/content-hub/mp-wiki/metrics/iou-intersection-over-union/601e415eb4-1653641012/9.png">

## Setup 

To use this algorithm, the repository should be locally cloned und die bash file setup.sh in the repository should be executed:

    git clone https://gitlab.com/link.developers.beta/ld-lib-ai-basics-py.git
    cd YOLO-ObjectDetection
    bash setup.sh 

The bash file will compile the necessary cython scripts in order to run them as a python library. After that, all dependencies for the use of this 
implementation from src will installed. In ./scr, the source code can be found. 

## To Run the project

### Alternative 1

To test the source code, you should navigate to the test file und run file test_Yolo_Model_Ev.py. 
    
    cd test
    pythom3.? test_Yolo_Mode_Ev.py

Subsequently, a interactive shell programm will be executed, where it could be specified which of the above explained actions should be executed.

### Alternative 2

Alternatively, you could use the implemented parser in the code und introduce after the file name which action you want to run. 
    
    pythom3.? test_Yolo_Mode_Ev.py --option=1 --file_path=./YOLO_Model_Evaluation/test/data_sample/ 

The option *_--option_* specified which option should be run:
    
    --option=1  --> YOLO Model Ev
    --option=2  --> generates a dataset with standard specifications
    --option=3  --> evaluate a already existing synthetic data

you can also specify the number of bounding boxes per image as well as the number of images, that would be using the same reference coordinates

    --num_bbox=3  
    --num_images=2
    
for options 1 and 3, you should also provide the path to the input data:
    
    --file_path=/path/to/archive/

for option 2, a path to the input data and the path to output file should be specified.
    
    --outfile=/path/to/output/file/

