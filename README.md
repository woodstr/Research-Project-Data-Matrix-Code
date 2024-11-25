* [Week 1](#week-1---3-october-2024)
* [Week 2](#week-2---10-october-2024)
* Week 3 (October Break)
* [Week 4](#week-4---24-october-2024)
* [Week 5](#week-5---31-october-2024-backup-used-due-to-travelling)
* [Week 6](#week-6---7-november-2024)
* [Week 7](#week-7---14-november-2024)
* [Week 8](#week-8---21-november-2024)
* [Week 9](#week-9---28-november-2024)
* [Week 10](#week-10---5-december-2024)
* [Week 11](#week-11---12-december-2024)

# Week 1 - 3 October 2024

## Goals

### Literature Review :heavy_check_mark:

Find and summarize some research papers (2 or 3 promising ones), about how to crop/rectify to reshape QR code and remove artifacts.
- https://arxiv.org/pdf/1506.02640
- https://iopscience.iop.org/article/10.1088/2631-8695/acb67e/pdf

Create a few slides for presenting the literature and what can be used.

https://slides.com/aidanstocks/literature-review

### Determine CNN Archetecture and Method to Adopt :heavy_check_mark:

Have some options, maybe with one I would prefer to pick. It needs to be:
- An architechture I understand
- Relatively small and efficient
- Easy to do transfer learning with

## Outcome of Week

Decided on:
- Trying out pretrained YOLO models on MAN ES provided dataset
- Finding a python library for non-rigid deformable transformation (bending cylinder shape etc.)
- Train non-pretrained YOLO on kaggle dataset to see how it performs
- See if YOLO pretrained on different datasets exists and compare

# Week 2 - 10 October 2024

## Goals

### Data Annotation :heavy_check_mark:

Annotate MAN ES provided dataset according to YOLO architecture. Model is predicting [x,y,W,h] of bounding boxes.

### YOLO models :heavy_check_mark:

Find (and download) some different pretrained YOLO models.

Also set up pipeline for training untrained YOLO models.

### Evaluation Metric :heavy_check_mark:

Decide upon an evaluation metric. Will likely be a simple metric like accuracy, as false positives should not be likely with Data Matrix decoding.

## Outcome of Week

### Data Annotation

Colleague showed me roboflow website. Used it to annotate the 180 images with the bounding boxes with no issues and high convenience (took maybe an hour).

### YOLO models

#### pretrained yolov11
The yolov11 model by [ultralytics](https://docs.ultralytics.com/) is pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset.

Without any additional training, it performed very poorly on the unseen MAN-ES datamatrix dataset. only 2 of the images had their DM codes located, with the classes "keyboard" and "book".

#### yolov11 trained on kaggle dataset
Training the yolov11 model from scratch on the [kaggle QR code dataset](https://www.kaggle.com/datasets/hamidl/yoloqrlabeled?resource=download).

#### pretrained yolov11 trained on kaggle dataset
Training the pretrained yolov11 model on the [kaggle QR code dataset](https://www.kaggle.com/datasets/hamidl/yoloqrlabeled?resource=download).

#### Fine-tuned versions of above
Same as above but fine-tuned to some of the DM codes from MAN.

## Evaluation Metric
Proposed evaluation metrics are:
- Rate of DM code locating (% of DM codes which had bounding boxes correctly around them)
- Rate of DM code decoding (% of DM codes which, when cropped to and fed to a standard decoder, yielded a decoded DM string)
- Rate of correct DM decoding (As above but confirmed correct)
- mAP50-95 of bounding boxes for validation data. ("The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95. It gives a comprehensive view of the model's performance across different levels of detection difficulty.")

# Week 3 (October Break)

# Week 4 - 24 October 2024

## Goals

### Models :heavy_check_mark:

This week will focus on performing the first experiment with the following models:
- yolov11 trained from scratch on kaggle dataset
- yolov11 trained from scratch on kaggle dataset and fine-tuned to MAN-ES DM dataset
- yolov11 pretrained by ultralytics on COCO dataset and fine-tuned on MAN-ES DM dataset

### MAN-ES Train/Val/Test Splits :heavy_check_mark:

MAN-ES test data will be changed in the following ways:
- Split into train/val/test with 100/30/50 splits.
- Train split will be augmented with rotations and sheering

### Decoding Pipeline :heavy_check_mark:

The decoding pipeline for making predictions, cropping to bounding boxes, and attempting to decode DM codes with pylibdmtx.

The pipeline will also calculate the following evaluations:
- DM decode rate (% of decodings of test images)
- Valid DM decode rate (% of _correct_ decodings of test images)
- mAP scores for bounding boxes (calculated by ultralytics, is essentially the accuracy of the predicted bounding boxes compared to the manually annotated one)

The outcome of this experiment will decide if we need to make any changes or if we can continue.

## Outcome of Week

### Model Performances
| Measure              | Baseline | Kaggle Scratch | Kaggle Finetuned  | Ultralytics Finetuned  |
| -------------------- | :------: | :------------: | :---------------: | :--------------------: |
| Precision            | N/A      | 0.25           | 0.91              | **0.96**               |
| Recall               | N/A      | 0.24           | 0.84              | **0.89**               |
| F1                   | N/A      | 0.25           | 0.87              | **0.92**               |
| mAP50-95             | N/A      | 0.069          | 0.746             | **0.753**              |
| DM decode rate       | 0.12     | 0.04           | 0.12              | 0.12                   |
| Valid DM decode rate | 0.12     | 0.04           | 0.12              | 0.12                   |
| Total Runtime (s)    | 153      | 34             | 40                | 43                     |

Notes:
- Runtime calculated on laptop
- Runtime was calculated as fair as reasonably possible (model pipeline stripped of most unnecessary calculations)
- Reason for different model decode speeds is likely due to increased false positives on finetuned models
- Baseline model being the slowest is likely due to there being many more pixels in the image compared to the cropped and resized images the models produce
- Kaggle Scratch fails to find 3 of the DM codes that other models find, leading to worse than baseline performance
- Somehow better decode rate on laptop? Need to look into / test this further

| Baseline | Kaggle Scratch | Kaggle Finetuned | Ultralytics Finetuned |
|:--------:|:--------------:|:----------------:|:---------------------:|
|<img width="200" alt="Baseline" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/baseline.png">|<img width="200" alt="Kaggle Scratch" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/Kaggle Scratch.png">|<img width="200" alt="Kaggle Finetuned" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/Kaggle Finetuned.png">|<img width="200" alt="Ultralytics Finetuned" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/Ultralytics Finetuned.png">|

# Week 5 - 31 October 2024 (backup used due to travelling)

# Week 6 - 7 November 2024

## Goals

### Minor Enhancement :heavy_check_mark:
Alter decoding pipeline to only crop and decode to bounding box with highest likelihood.

### Statistics of Distortion MAN dataset :heavy_check_mark:
From the decoder outputs, make some statistics on the types of dmc's that succeed and fail. Steps:
 - Run decoder to get failure cases
 - Visually compare and try to find similarities in the failure cases compared to successful
 - Which factors are important? Color? Distortion? Reflections?

This is relevant for deciding which method to use in next experiment.

Note for possible data synthesis:
 - Ground truth = true positive, we want to synthesize the negative cases! This way we can train a model to convert the negative cases to positive.

### Research :heavy_check_mark:
Check papers on rectification and image binarization. (Look for high citations, and prioritize image binarization).

There are some papers already doing both (Google "qr code binarization").

Summarize them.

## Outcome of Week

### Minorly Improved Model Performances
| Measure              | Baseline | Kaggle Scratch | Kaggle Finetuned  | Ultralytics Finetuned  |
| -------------------- | :------: | :------------: | :---------------: | :--------------------: |
| Precision            | N/A      | 0.25           | 0.91              | **0.96**               |
| Recall               | N/A      | 0.24           | 0.84              | **0.89**               |
| F1                   | N/A      | 0.25           | 0.87              | **0.92**               |
| mAP50-95             | N/A      | 0.069          | 0.746             | **0.753**              |
| DM decode rate       | 0.12     | 0.04           | 0.12              | 0.12                   |
| Valid DM decode rate | 0.12     | 0.04           | 0.12              | 0.12                   |
| Total Runtime (s)    | 173      | 25             | 40                | 31                     |

Notes:
- Tests run similar to previous.
- Minor changes in baseline likely due to randomness in background processes.
- Models perform faster than previous test by only decoding high confidence bounding box. Note that this is even with slower laptop performance.
- "Ultralytics Finetuned" benefitted greater than "Kaggle Finetuned"

### Statistics of Distortion MAN dataset
Relevant statistics of failure cases of baseline decoder shown below.
| Most common color combinations (front/back) | Lazer vs. Dot | Types of Distortions |
| :-----------------------------------------: | :-----------: | :------------------: |
|<img width="500" alt="Color Combo" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/stats_color_combo.png">|<img width="500" alt="Lazer vs. Dot" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/stats_lazer_dot.png">|<img width="500" alt="Distortion Types" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/stats_malformation_types.png">|

Note on success cases:
- Either black on metal or white on green
- All lazer
- Minor blur
- Some are small
- One example with high contrast change is decoded

Conclusion: dot malformation and blur are the main factors for failure. Focus on either of these could produce much better results.

### Research
Here I've listed options good for different reasons

#### Image Binarization
Image binarization should greatly reduce high contrast change (9 images) and possibly cover and blur (23+ images).

| Title                                                                                   | Citations | Link                                                  |
| --------------------------------------------------------------------------------------- | :-------: | :---------------------------------------------------: |
| Adaptive Binarization of QR Code Images for Fast Automatic Sorting in Warehouse Systems | 33        | [link](https://www.mdpi.com/1424-8220/19/24/5466/pdf) |
| Fast Adaptive Binarization of QR Code Images for Automatic Sorting in Logistics Systems | 4         | [link](https://www.mdpi.com/2079-9292/12/2/286/pdf)   |

Image binarization is the process of converting pixel values to either black or white. This could be very useful for DMC detection as it could isolate the pixel values containing the DMC from other irrelevent pixel values, potentially increasing the performance of the baseline decoder.

Both papers relate to each other (one builds on the other), and the code is not available, so this method may prove hard to use. However the method runs very fast (0.04s / 25 FPS) and seems to perform very well for improving QR code decode rates.

#### Blur Removal Methods
Focusing on blur removing methods could greatly reduce the problem of blur present in 22/50 of the test dataset.

| Title                                                            | Citations | link                                                            |
| ---------------------------------------------------------------- | :-------: | :-------------------------------------------------------------: |
| Fast Blur Removal for Wearable QR Code Scanners                  | 26        | [link](https://files.ait.ethz.ch/projects/quick-qr/quickQR.pdf) |
| DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better | 1086      | [link](https://arxiv.org/pdf/1908.03826)                        |

Blur removal methods seem to typically involve using models to try and estimate how to de-blur an image. The main theory behind it is that since algorithms can blur images, they can also be used to deblure images. However because of noise in real images, you need to estimate the deblurring. Models can be used to estimate the best methods for deblurring a given image. The above two papers cover such models.

The fast blur removal method combines a few seemlingly very specific methods together, and there is no code publicly available, so I do not favor this option.

"DeblurGAN-v2 is based on a relativistic conditional GAN with a doublescale discriminator". More on it in the proposal below.

#### Proposal
I propose going primarily with the DeblurGAN-v2 paper/model. Reasons:
- [The code is freely available](https://github.com/VITA-Group/DeblurGANv2?tab=readme-ov-file)
- The model performs at real-time speed (0.06s / ~17FPS)
- From my human perspective the model seems to deblur [pretty well](https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/refs/heads/master/doc_images/restore_visual.png). [Extra example](https://github.com/VITA-Group/DeblurGANv2/blob/master/doc_images/kohler_visual.png).
- A [paper](https://arxiv.org/pdf/2109.03379) used a [similar model](https://github.com/York-SDCNLab/Ghost-DeblurGAN?tab=readme-ov-file) for [detecting a different kind of code](https://user-images.githubusercontent.com/58899542/154817295-22e733a5-5f33-439d-a29e-08f5950a8784.gif).

[DeblurGAN-v2 architecture](https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/refs/heads/master/doc_images/pipeline.jpg).

#### Actual Decision
Final decision for step 2 is very different than proposal!

Main idea: train a simple image binarizer on a synthesized dataset. More details on process in next weeks plan.

# Week 7 - 14 November 2024

## Goals

### Experiment 2 :heavy_exclamation_mark:

This week will focus on getting step 2 of the overall process done: an image binarizer.

#### Image Binarizer :heavy_exclamation_mark:
By training an image binarizer we can reduce many issues prevalent in the real-world images. Issues such as:
- blur
- contrast difference
- other color related issues

Notes:
- Issues relating to dot-markings will be ignored for the paper, and be reserved for future MSc. project.
- Issues relating to rectification will be ignored on the basis that the baseline decoder has it's own rectification methods, which may be sufficient if we remove the other issues.
- Will consider finding a small pretrained resnet-18 model for finetuning on a dataset. If impossible train from scratch.
- May have to customize last layer, it should not have a classification head but be a 1 channel binary map with range of 0-1.
  - Last layer of e.g. sigmoid function can squeeze to 0-1.

#### Dataset Synthesis :heavy_check_mark:
In order to train an image binarizer we need to have an appropriate dataset. By generating DM codes and blending onto metal backgrounds with different augmentations, we may be able to train a binarizer to effectively reduce much of the noise present in the real-world pictures and improve the decode-rate of the baseline decoder.

The proposed steps for generating a dataset is as follows:
1. Generate list of random strings to encode
2. Generate DMCs from strings
3. Apply shape transformations to DMCs (affine, rotation, flip, crop, ...)
   - Save for later
4. Scale DMCs randomly between 0-1
5. Select random crops of random metal textures
6. Blend DMCs to texture crops (multiply scaled DMCs with backgrounds)
   - Stamp saved shape-transformed DMCs onto white background of same size as texture crops for use as ground truths
7. Apply color transformations to blended images (color jitter, contrast, ...)

The steps should effectively give us a dataset with two different groups of images:
1. **Ground truth images**. These are the images that we want to the image binarizer to create (black DMC on white background). Image binarizers are effective at removing certain distortions like color and blur, which is why the ground truth images still contain shape distortions.
2. **Synthesized images**. These are the images that we wish to teach the binarizer to transform. They should be similar to the real-world images, and the model should learn how transform them into the ground truth images.

Tips for later:
- [pylibdmtx](https://pypi.org/project/pylibdmtx/) can do DMC generation
- Relevant metal textures can be found [online](https://www.google.com/search?q=machined+metal+texture).
  - Textures with different areas could be good when combined with the random cropping.
- For transformations, [pytorch has many methods](https://pytorch.org/vision/main/transforms.html#v2-api-reference-recommended) that can be used.
- Ensure to define the two groups of transformations well in paper! (e.g. shape transformation is consistent from input-output but color transformation not)
- Ensure input/output sizes are the same! Output of step 1 has a specific shape so use that.
- DO NOT RETRAIN STEP 1 ON NEW DATA! It does not reduce decode-rate, so we can safely assume it performs well. Later on if time allows consider retraining, as having less training data sources simplifies the code.

## Outcome of Week

#### Dataset Synthesis

Synthesized 10,000 noisy and ground truth image pairs according to specs, some examples below.

|              | Example 1 | Example 2 | Example 3 |
| :----------: | :-------: | :-------: | :-------: |
| Noisy        | <img width="500" alt="noisy_0" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/noisy_0.png"> | <img width="500" alt="noisy_1" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/noisy_1.png"> | <img width="500" alt="noisy_2" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/noisy_2.png"> |
| Ground Truth | <img width="500" alt="ground_truth_0" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/ground_truth_0.png"> | <img width="500" alt="ground_truth_1" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/ground_truth_1.png"> | <img width="500" alt="ground_truth_2" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/ground_truth_2.png"> |

#### Image Binarizer
The Image Binarizer structure is set up, but it needs to be further debugged and tweaked to improve its results. Currently removes all noise completely and shows a black blob where the DMC is.

| Noisy (ResNet Normalized) | Ground Truth | Predicted |
| :-----------------------: | :----------: | :-------: |
| <img width="500" alt="failure_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/failure_noisy.png"> | <img width="500" alt="failure_ground_truth" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/failure_ground_truth.png"> | <img width="500" alt="failure_prediction" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/failure_prediction.png"> |

# Week 8 - 21 November 2024

## Goals

### New Step 2 Method :heavy_exclamation_mark:
Instead of ResNet method, will try the following options:
#### Histogram Analysis :heavy_check_mark:
From the output of YOLO models, I can produce histograms of the images (grayscale value on x-axis and count on y).

I can create figures of all MAN test images side by side with their histogram plots.

From manually inspecting all of these figures, I can see if I am very lucky and there is a common threshold to use for binarizing all of the images.

If a common threshold exists, the second step can be this binarization process, and the baseline decoder can run on both sides of the binarization.

#### Prerequisites for Later Methods :heavy_check_mark:
Later methods require the following fixes:
- Fix pipeline to use YOLO cropped images
- Fix blending method used in data synthesis to use multiplication method. Different values of blackness for the DMC can be used to have the blended image have a darker or more faded DMC
  - Ensure there are no ultra faded DMCs this time, as they do not represent real-world data 

#### U-Net
Instead of using ResNet, use U-Net to predict DMC pixels apart from background pixels:

Apply Z-Score normalization (either across all 3 channels if RGB or 1 channel if grayscale)

Try with and without Z-Score...

#### U-Net with Histogram
If the above U-Net performs poorly, remove Z-Score and calculate threshold instead.

The mask from U-Net output can be used to find the DMC pixel areas from the OG image. With those pixels from the OG image, we can create a histogram of it to calculate a threshold to use for binarization.

#### Predicting Threshold
If all else fails, try this hail mary.

Return to using ResNet, but add a small CNN head instead, with the purpose of predicting the threshold to use in binarization using the output of ResNet as input.

This relies on there being valid thresholds to use, but may work well.

#### IF ALL ELSE FAILS :heavy_check_mark:
If all else fails I can return to original ResNet. Maybe the fixes will improve it 🥲. Can also try tweaking values and calculating weight for BCE loss to deal with imbalance in class labels.

## Outcome of Week

### Histogram Analysis
Initial analysis of histogram analysis shows that many images do have two peaks of grayscale values, where binarizing with a threshold in between effectively binarizes the image. However, while there are common thresholds for different groups of images, there is no global common threshold.

Binarizing with Otsu's method (a method that calculates the threshold of a given image), does effectively binarize most of the images. However, noisy images (blurry, high contrast), still fail to be decoded. It seems that these noise factors need to be reduced, and pure binarization is not an effective method for increasing decode rate. Examples below.

| success | failure |
| :-----: | :-----: |
| <img width="500" alt="success_0" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/histogram_binarization/success_0.png"> | <img width="500" alt="failure_0" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/histogram_binarization/failure_0.png"> |
| <img width="500" alt="success_1" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/histogram_binarization/success_1.png"> | <img width="500" alt="failure_1" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/histogram_binarization/failure_1.png"> |

Not that my default phone camera can decode many of the images when I point it at my computer screen. Perhaps the baseline decoder performs very poorly itself, and a method that optimizes for the decoder would be preferable.

Using Otsu's method gave the same decode rate as baseline, 0.12.

### Return to ResNet
I decided on returning to ResNet instead of exploring U-Net, as some minor fixes led to a model able to create some binarized images that could be decoded:

| Example 1 | Example 2 | Example 3 |
| :-------: | :-------: | :-------: |
| <img width="500" alt="decodable_0" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/decodable_0.png"> | <img width="500" alt="decodable_1" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/decodable_1.png"> | <img width="500" alt="decodable_2" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/binarization/decodable_2.png"> |

However, these results were not easily reproduceable. I believe I added too much variance to the data synthesis, to where the images are no longer representative of real-world data. Redoing the synthesis and trying again is an option, but the Otsu binarization gave more robust results.

Using ResNet gave a worse decode rate, 0.6.

# Week 9 - 28 November 2024

## Goals

### Final coding :on:

Yucheng has spotted mistakes I have made in the code. This final week will be spent with rewrites and fixes based on his advice.

#### More Textures :heavy_check_mark:

I should get around 50-100 different metal textures as backgrounds for the DMCs.

#### Code fixes :heavy_check_mark:
- Don't normalize images at any stages
- Early stopping based on validation, not training!
- Change augmentation based on notes
- Swap from ResNet to U-Net

### Initial Report Writing :on:
Veronika will be reading research paper reports on 5th December. Start writing and be sure to submit before then.

## Outcome of Week

### More Textures
Spent a couple hours getting free metal textures. Now have 53.

### Final Model Performance
With the final code fixes and changes, we see an improvement in performance! I trained 3 different models. One that relies fully on BCE loss, another fully on Dice loss, and a final one using a 1:1 mix of the two.

Below is a Table showing YOLO crops and the binarized image from successful decodings of each Model. The model relying purely on Dice loss produced purely white images, so it is excluded.

| YOLO Crop | BCE Loss Binarizer | BCE + Dice Loss Binarizer |
| :-------: | :----------------: | :-----------------------: |
| <img width="500" alt="1_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/1_noisy.png"> | <img width="500" alt="100_1_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_1_bin.png"> | N/A |
| <img width="500" alt="2_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/2_noisy.png"> | <img width="500" alt="100_2_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_2_bin.png"> | <img width="500" alt="102_2_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/102_2_bin.png"> |
| <img width="500" alt="3_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/3_noisy.png"> | <img width="500" alt="100_3_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_3_bin.png"> | <img width="500" alt="102_3_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/102_3_bin.png"> |
| <img width="500" alt="4_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/4_noisy.png"> | <img width="500" alt="100_4_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_4_bin.png"> | <img width="500" alt="102_4_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/102_4_bin.png"> |
| <img width="500" alt="5_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/5_noisy.png"> | <img width="500" alt="100_5_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_5_bin.png"> | <img width="500" alt="102_5_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/102_5_bin.png"> |
| <img width="500" alt="6_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/6_noisy.png"> | <img width="500" alt="100_6_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_6_bin.png"> | <img width="500" alt="102_6_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/102_6_bin.png"> |
| <img width="500" alt="7_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/7_noisy.png"> | <img width="500" alt="100_7_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_7_bin.png"> | <img width="500" alt="102_7_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/102_7_bin.png"> |
| <img width="500" alt="8_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/8_noisy.png"> | <img width="500" alt="100_8_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_8_bin.png"> | <img width="500" alt="102_8_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/102_8_bin.png"> |
| <img width="500" alt="9_noisy" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/9_noisy.png"> | <img width="500" alt="100_9_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/100_9_bin.png"> | <img width="500" alt="102_9_bin" src="https://github.com/woodstr/Research-Project-Data-Matrix-Code/blob/main/figures/final_binarization/102_9_bin.png"> |

Below is a final Table of performance of decode rates.

| Measure              | Baseline | YOLO   | YOLO + Binarizer |
| -------------------- | :------: | :----: | :--------------: |
| Valid DM decode rate | 0.12     | 0.12   | **0.18**         |
| Total Runtime (s)    | 46       | **27** | 148              |

Notes:
- Experiment done on laptop cpu.
- YOLO gives the best speed, without sacrificing decode rate.
- While YOLO + Binarizer brings a better decode rate (over baseline!), it slows down performance to less than real-time. (~3s to process each image)

# Week 10 - 5 December 2024

## Goals

Report Writing.

## Outcome of Week



# Week 11 - 12 December 2024

### Finish Paper :x:

Paper is due 16th December at 14:00.
