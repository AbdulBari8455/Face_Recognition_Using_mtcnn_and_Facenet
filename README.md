# Face_Recognition_Using_mtcnn_and_Facenet

#MTCNN:
Perform multiple tasks Cascaded CNN is a cutting-edge instrument for face location, 
utilizing a 3-stage neural organization locator.

![image](https://user-images.githubusercontent.com/73050746/147135287-9e48cb6e-6a11-4350-92f3-b0c80bbbafc3.png)

First, the image is resized multiple times to detect faces of different sizes. Then the P-network (Proposal) scans images, performing first detection. It has a low threshold for detection and therefore detects many false positives, even after NMS (Non-Maximum Suppression), but works like this on purpose.
The proposed regions (containing many false positives) are input for the second network, the R-network (Refine), which, as the name suggests, filters detections (also with NMS) to obtain quite precise bounding boxes.
The final stage, the O-network (Output) performs the final refinement of the bounding boxes. This way not only faces are detected, but bounding boxes are very right and precise.
An optional feature of MTCNN is detecting facial landmarks, i.e. eyes, nose and corners of a mouth. It comes at almost no cost, since they are used anyway for face detection in the process, which is an additional advantage if you need those (e.g. for face alignment).
The official TensorFlow implementation of MTCNN works well, but the PyTorch one is faster (link). It achieves about 13 FPS on the full HD videos, and even up to 45 FPS on rescaled, using a few tricks (see the documentation). It’s also incredibly easy to install and use. I’ve also achieved 6–8 FPS on the CPU for full HD, so real-time processing is very much possible with MTCNN.
MTCNN is very accurate and robust. It properly detects faces even with different sizes, lighting and strong rotations. It’s a bit slower than the Viola-Jones detector, but with GPU not very much. It also uses color information, since CNNs get RGB images as input.


