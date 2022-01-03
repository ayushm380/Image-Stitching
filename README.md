# Image-Stichting
Performed feature detection to find keypoints (SIFT, SURF, ORB) used ORB. Then matching keypoints (knn, brute force) used brute force. Used RANSAC to estimate homography which transforms one key points to other. Used this homography to warp the second image in persepective of first. Then pasted both images together. At end used histogram equilization for seam removal.
# Sample Outputs 
First Image
![1](https://user-images.githubusercontent.com/75737493/147912999-3bda2966-b0b0-4af1-bf93-5b61f35a0ad9.png)
Second Image
![2](https://user-images.githubusercontent.com/75737493/147913008-2c6eb0ef-a60c-4e16-9e2f-55334d53d04e.png)
Output image
![output](https://user-images.githubusercontent.com/75737493/147913028-2edaf498-5f62-453b-828d-f4835f17ac17.png)
