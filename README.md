# 4dsight-Challenge

P.S. This project made for a company's job application challenge project! Not personal project! <br /> 

<br /> <br /> 
Output videos were generated using three different state-of-the-art video object segmentation solutions such as Faster R-CNN with MobileNetV3, YoloV3, and Mask R-CNN with ResNet50 backbone. These models are relatively old models but personally I wanted to chose them to make challenge project explanations much more simple and understandable.
These solutions were applied to three sample videos: "video_sample1", "video_sample2", and "video_sample3". The objective was to evaluate the segmentation and tracking results, focusing on moving objects, particularly horses. 
<br /> <br /> <br /> 
Clarity of Segmentation
The segmentation results provided by YoloV3 were generally clear and accurate across all sample videos. Objects, including horses, were well-defined and distinguishable from the background. Much more smoother and accurate. Mask R-CNN performed better than MobileNetV3(Faster RCNN), although not as well as Yolov3. While Faster R-CNN with MobileNetV3 produced acceptable segmentation in some cases, it occasionally struggled with accurately delineating object boundaries, resulting in less clear segmentation compared to the other models.
<br /> <br /> <br /> 
Evaluation and Reasoning &Choice of Solutions
Yolov3 was chosen for its excellent balance of speed and accuracy in object detection and tracking tasks. It demonstrated consistent performance across different scenes and conditions, making it a reliable choice for video object segmentation. Mask R-CNN was selected for its superior ability to generate precise segmentation masks, which proved beneficial for accurately delineating object boundaries, particularly in cluttered scenes. Although Faster R-CNN with MobileNetV3 is known to give successful results in real-time applications, I think it gave poor results. It was so surprising.
<br /> <br /> <br /> 
Strengths and Weaknesses
YoloV3, relatively, fast processing speed, robust object detection, and reliable tracking capabilities. Mask R-CNN, has, precise segmentation, accurate object tracking, and resilience to background clutter. Higher computational overhead and slower processing speed compared to Yolov3, are, the some of the weaknesses. Faster R-CNN, has, fast inference speed and lightweight architecture suitable for real-time applications. Lower segmentation accuracy and tracking consistency compared to YOLOv3 and Mask R-CNN are the weaknesess.
<br /> <br /> <br /> 
Performance Evaluation
YoloV3 and Mask R-CNN consistently outperformed Faster R-CNN with MobileNetV3 in terms of segmentation clarity and tracking accuracy across all sample videos. YoloV3, in particular, demonstrated superior performance in challenging scenarios, making it the preferred choice for video object segmentation tasks involving horses. MobileNetV3 offered acceptable performance in some cases (for video_sample2,
video_sample3), its overall performance was less consistent and reliable compared to Yolov3 and Mask R-CNN, highlighting its limitations in handling complex scenes and rapid object movements.
<br /> <br /> <br /> 
Additional Thoughts
Various post-processing techniques can be used to improve model outputs. For example, we can make object boundaries smoother or reduce noise by using techniques such as non-maximum suppression (This has been done, but more time needs to be spent on this solution), anti-aliasing or pixel merging. More powerful hardware (for example, faster GPUs) can be used to increase the performance of the model.
As mentioned earliar, other State-of-Art Models (e.g. upper versions of YoloV4-V8, or SSD-Single Shot MultiBox Detector) can be used to take much more better results.
<br /> <br /> <br /> 

MobileNet-ResNet-YoloV3 (for video_sample1) : <br /> <br /> 

![image](https://github.com/JiyuuX/4dsight-Challenge/assets/139239394/4185e770-18bd-42e9-916e-dd5825ca8b71)

<br /> <br /> <br /> 
MobileNet-ResNet-YoloV3 (for video_sample2) : <br /> <br /> 

![image](https://github.com/JiyuuX/4dsight-Challenge/assets/139239394/d8749045-5bf9-4831-b3ca-5081ba820caa)

<br /> <br /> <br /> 
MobileNet-ResNet-YoloV3 (for video_sample3) : <br /> <br /> 

![image](https://github.com/JiyuuX/4dsight-Challenge/assets/139239394/bde8d9c4-8dfb-4960-a6c3-40b92d1c4076)

