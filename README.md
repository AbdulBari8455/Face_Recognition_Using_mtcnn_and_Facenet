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

#FaceNet

Face Recognition is becoming a new trend in the security authentication systems. Modern FR systems can even detect, if the person is real(live) or not while doing face recognition, preventing the systems being hacked by showing the picture of a real person. I am sure, everyone wondered when Facebook implemented the auto-tagging technique. It identifies the person and tag him/her when ever you upload a picture. It is so efficient that, even when the person’s face is occluded or the picture is taken in darkness, it tags accurately. All these successful face recognition systems are the results of recent advancements in the field of computer vision, which is backed by powerful deep learning algorithms. Let us explore one of such algorithms and see how we can implement a real time face recognition system.
Face recognition can be done in two ways. Imagine you are building a face recognition system for an enterprise. One way of doing this is by training a neural network model (preferably a ConvNet model) , which can classify faces accurately. As you know for a classifier to be trained well, it needs millions of input data. Collecting that many images of employees, is not feasible. So this method seldom works. The best way of solving this problem is by opting one-shot learning technique. One-shot learning aims to learn information about object categories from one, or only a few, training images. The model still needs to be trained on millions of data, but the dataset can be any, but of the same domain. Let me explain this more clearly. In one shot way of learning, you can train a model with any face datasets and use it for your own data which is very less in number. 
One-shot learning can be implemented using a Siamese network. As the name indicates, its nothing but, two identical neural networks with exact same weights, but taking two distinct inputs. These networks are optimised based on the contrastive loss between their outputs. This loss will be small when the inputs to the networks are similar and large when inputs differ from each other. So in this way, optimised Siamese networks can differentiate between their inputs.
![image](https://user-images.githubusercontent.com/73050746/147217188-51793827-909c-4114-b0e7-4d46a76c8894.png)
FaceNet is a one-shot model, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors(from the original paper).To train, they used triplets of roughly aligned matching / non-matching face patches. A triplet is nothing but a collection one anchor image, one matching image to the anchor image and one non-matching image to the anchor image. So the triplet loss minimises the distance between an anchor and a positive, both of which have the same identity, and maximises the distance between the anchor and a negative of a different identity.
![image](https://user-images.githubusercontent.com/73050746/147217333-d2ab22ed-851a-4796-ac50-3e48cd420b8f.png)

Collect the images of all Persons.

Align the faces using MTCNN (Multi-task Cascaded Convolutional Neural Networks), dlib or Opencv. These methods identify, detect and align the faces by making eyes and bottom lip appear in the same location on each image.

Use the pre-trained facenet model to represent (or embed) the faces of all employees on a 128-dimensional unit hyper sphere.

Store the embeddings with respective Persons names on disc.

Now Train an ML algorithm on these embeddings.

Now your face recognition system is ready !!. Let us see how we can recognise faces, with what all we have done above. Now you have with you, the corpus of 128-dimensional embeddings with corresponding employee names. When ever an employee faces your detection camera, the image being captured will be ran through the pre-trained network to create the 128-dimensional embedding which will then be compared to the stored embeddings using euclidean(L2) distance. If the lowest distance between the captured embedding and the stored embeddings is less than a threshold value, the system can recognise that person as the employee corresponding to that lowest distant embedding.

Align Dataset Accordingly For Training:
Training Dataset

|...Person1

|...Person2

|...Person3

|...Person4

That’s all…. You have built a simple, but efficient face recognition system. Have a nice day…!!
