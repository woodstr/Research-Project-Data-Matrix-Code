* [Week 1](#week-1---3-october-2024)
* [Week 2](#week-2---10-october-2024)
* Week 3 (October Break)
* [Week 4](#week-4---24-october-2024)
* [Week 5](#week-5---31-october-2024)
* [Week 6](#week-6---7-november-2024)
* [Week 7](#week-7---14-november-2024)
* [Week 8](#week-8-backup---21-november-2024)
* [Week 9](#week-9---28-november-2024)
* [Week 10](#week-10-backup---5-december-2024)
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

### Research :on:
Check papers on rectification and image binarization. (Look for high citations, and prioritize image binarization).

There are some papers already doing both (Google "qr code binarization").

Summarize them.

### Statistics of Distortion MAN dataset :on:
From the decoder outputs, make some statistics on the types of dmc's that succeed and fail. Steps:
 - Run decoder to get failure cases
 - Visually compare and try to find similarities in the failure cases compared to successful
 - Which factors are important? Color? Distortion? Reflections?

This is relevant for deciding which method to use in next experiment.

Note for possible data synthesis:
 - Ground truth = true positive, we want to synthesize the negative cases! This way we can train a model to convert the negative cases to positive.

## Outcome of Week

### New Model Performances
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
- Minor changes in baseline due to random background processes.
- Models perform faster by only decoding high confidence bounding box!

# Week 7 - 14 November 2024

## Goals

### Experiment 2 :x:

This week will focus on performing the second experiment with any new models / additions to the previous pipeline.

## Outcome of Week

TBD

# Week 8 - 21 November 2024

## Goals

### Report Writing :x:

Bulk writing of report.

## Outcome of Week

TBD

# Week 9 - 28 November 2024

## Goals

### Initial Report Due :x:

Veronika will be reading research paper reports on 5th December. Submit initial version of report before then so that she can read and give feedback.

## Outcome of Week

TBD

# Week 10 (Backup) - 5 December 2024

## Goals

Report Writing.

## Outcome of Week



# Week 11 - 12 December 2024

### Finish Paper :x:

Paper is due 16th December at 14:00.
