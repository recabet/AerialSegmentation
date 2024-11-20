# Roof Segmentation from Aerial Images

## Data Preprocessing
The data labels were divided into two categories: one with the keyword "vis" and one without. Labels without "vis" were all black, so I removed them from the dataset. After this step, the data contained 732 training images and corresponding masks. Since nearly 20% of the training images had no roofs, I decided to remove them to save computational resources and help the model learn better, as the number of white pixels (representing roofs) was already very small compared to the black pixels.

I reduced the image size from (10000, 10000) to (128, 128), which may lose some details but greatly reduces training time. I decided to normalize the training images by dividing them by 255, which helps the model converge to optimal weights faster.

I used the `resplit_dataset` function defined in `resplit.py` to split the data into 70/20/10 as requested in the assessment.

I used the `load_images_masks` function defined in `load_images.py` to load the data as a `np.array`.

## Model
I decided to use the U-Net model architecture as it is well-suited for segmentation tasks. U-Net is a deep learning architecture designed for image segmentation tasks, especially in scenarios with limited data. It features an encoder-decoder structure with skip connections that preserve spatial information during downsampling and upsampling.

I called the model with the `unet_model` function defined in `model.py`.

I compiled the model using the `adam` optimizer and used `binary-crossentropy` as the loss function.

I chose `batch_size=1` and `epochs=10` because, from experience, CNN-based models work better with smaller batch sizes. The training results did not improve significantly after the 10th epoch.

## Challenges
1. **Computational Resources**: Training times were long, consuming a lot of CPU, GPU, and RAM.
2. **Overshooting**: Depending on the initial weights, the model sometimes overshot the minimum during training.
3. **Hyperparameter Tuning**: Hyperparameter tuning was challenging due to the long training times.
4. **Class Imbalance**: The ratio of black pixels to white pixels was large, making it harder for the model to learn effectively.

## Results
The model learned most of the patterns, as shown by the high training IoU and above-average test IoU. The visualizations of some predictions and ground truths were satisfactory. The F-1 score, Accuracy, Recall, and Precision scores were also good. Based on experience, an IoU score above 60 is usually considered satisfactory for segmentation tasks. The model achieved approximately 72 IoU, which is a good result.

Overall, I would classify the model as "mostly learned." Pixel-wise scores were also calculated and showed similar results to the aggregate training and test scores.
