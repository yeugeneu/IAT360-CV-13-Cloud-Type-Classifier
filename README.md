# Cloud Spotter ☁️
**IAT 360 – Exploring Artificial Intelligence · Fall 2026 · Computer Vision Project · Group 13**

Martina Hokama · Marielle Reyes Lao · Dalena Nguyen · Eugene Yu

Cloud Spotter detects and names cloud types in ground-level photos of the sky. A user takes a photo with their phone, the model draws a box around each cloud and labels its WMO cloud type, and an app could then explain what that cloud usually means for the weather in the next few hours (nowcasting).

- **Task:** supervised object detection
- **Model:** YOLOv8s (Ultralytics), fine-tuned from pretrained `yolov8s.pt`
- **Classes (11):** the 10 WMO cloud genera plus contrails
- **Users:** people who are curious about the sky (walkers, hikers, commuters), using a smartphone

Report: https://1sfu-my.sharepoint.com/:w:/g/personal/yya6_sfu_ca/IQBUeCsJ7HNUQqifsrDW6m_zAUA4W7plNKpYUp22v9rOC0U?e=UmFaam

🎥 Presentation video: [Presentation Video Link](https://1sfu-my.sharepoint.com/:v:/g/personal/yya6_sfu_ca/IQBk6nJrRptiQKmjsSfZ1hisAYmS7mes7kNQveTg42koDYs?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=VxxrII)

---

## Classes

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | Altocumulus | 6 | Cumulonimbus |
| 1 | Altostratus | 7 | Cumulus |
| 2 | Cirrocumulus | 8 | Nimbostratus |
| 3 | Cirrostratus | 9 | Stratocumulus |
| 4 | Cirrus | 10 | Stratus |
| 5 | Contrails | | |

Label format (YOLO): one line per box, `class x_center y_center width height`, with coordinates normalized to 0–1.

---

## Dataset

| Split | Images | Share | Source |
|---|---|---|---|
| Train | 2,016 | 76.3% | Roboflow (CCSN photos) |
| Validation | 505 | 19.1% | Roboflow (CCSN photos) |
| Test | 120 | 4.5% | Our own data |
| **Total** | **2,641** | | |

**Training and validation data.** The [Roboflow "Cloud" dataset (v1)](https://universe.roboflow.com/ip2-kbjz5/cloud-qthye/dataset/1) (CC BY 4.0), which contains the [CCSN](https://github.com/upuil/CCSN-Database) cloud photos with bounding boxes. We forked it and exported a version without Roboflow's augmentations (rotated, upside-down and grayscale copies), then removed photos duplicated across splits, leaving 2,521 unique images. Split 80/20 into train and validation (`random_state=42`).

**Test data (our own).** 120 images, 30 per member: 91 of our own phone photos, 20 web images and 9 images from the [Kaggle Cloud Types dataset](https://www.kaggle.com/datasets/newgnart/cloud-types) (MIT licence). Annotated with bounding boxes in [Label Studio](https://labelstud.io) and exported in YOLO format.

> ⚠️ **Our test images are not stored on GitHub.** Some photos show people and the windows of homes, so for privacy they are shared only as a zip.
> **Download [Members_Dataset.zip](https://1sfu-my.sharepoint.com/:u:/r/personal/yya6_sfu_ca/Documents/CV%20Project%20Group%2013/Project%20Submission%20Files/Members_Dataset.zip?d=w46167006cbbe43928c0f9c271bec379d&csf=1&web=1&e=m5VUUH) and extract it into `All_data/`.** It contains `images/test/` and `labels/test/`, matching the empty folders in this repository. The images were re-saved during preprocessing, so they contain no location (EXIF) data.

---

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/yeugeneu/IAT360-CV-13-Cloud-Type-Classifier.git
   ```
2. Download `Members_Dataset.zip` (link above) and extract it into `All_data/`.
3. Open `Cloud_type_detection.ipynb` in Google Colab and select a GPU runtime (**Runtime → Change runtime type → T4 GPU**).
4. Upload the repository folder to Google Drive (we used `MyDrive/IAT 360/`), or change the paths at the top of the notebook to where you put it.
5. Run the cells in order. To skip training, load the final model directly:
   ```python
   from ultralytics import YOLO
   model = YOLO("weights/baseline_best.pt")
   model.predict("your_sky_photo.jpg", save=True)
   ```

---

## Repository structure

```
├── Cloud_type_detection.ipynb   # full pipeline: preprocessing → training → evaluation → tuning
├── config.yaml                  # YOLO data config (relative paths)
├── All_data/
│   ├── images/{train,val,test}  # test/ is empty: extract the zip here
│   └── labels/{train,val,test}
├── weights/
│   └── baseline_best.pt         # final selected model
└── screenshots/                 # screenshots of each step
```

---

## Results

**Final model: the baseline** (YOLOv8s, 50 epochs, YOLO defaults: AdamW, lr0 = 0.000667, image size 640).

| Set | Precision | Recall | F1 | Accuracy* | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Validation | 0.439 | 0.607 | 0.509 | 0.535 | 0.536 | 0.468 |
| Test (our photos) | 0.359 | 0.185 | 0.244 | 0.190 | 0.111 | 0.056 |

*Object detection has no standard accuracy metric, so we define accuracy as the share of ground-truth clouds that the model detected and labelled with the correct type (missed clouds count as wrong).

- Best class: **Cumulonimbus** (mAP50 0.793). Weakest: **Stratus** (0.224), often confused with stratocumulus.
- The large drop on our test photos comes mainly from the difference between the clean online training images and real phone photos (buildings, wires, mixed skies), and from different labelling styles.
- A model combining every member's best hyperparameter (Adam, lr0 = 0.0001, box = 12, weight decay = 0.05) reached mAP50 0.516, below the baseline, so the baseline was selected.

---

## Hyperparameter tuning

Each member changed one hyperparameter to three values. Because of Colab GPU limits, every run (including a control with the baseline's setting) was trained for 15 epochs from `yolov8s.pt`, with everything else unchanged. The best values were then combined and trained for 50 epochs.

| Member | Hyperparameter | Values tested (control) | Best (validation) |
|---|---|---|---|
| Martina Hokama | Learning rate `lr0` (optimizer set to AdamW) | 0.0001, 0.001, 0.01 (0.000667) | 0.0001 |
| Dalena Nguyen | Optimizer | SGD, Adam, RMSProp (AdamW) | Adam |
| Marielle Reyes Lao | Weight decay | 0, 0.005, 0.05 (0.0005) | 0.05 |
| Eugene Yu | Box loss gain `box` | 3.0, 12.0, 20.0 (7.5) | 12.0 |

Each member's runs and outputs should be in the main branch notebook under these sections: 
* Optimizer Hyperparameter (Dalena)
* Learning Rate Hyperparameter (Martina)
* Weight Decay Hyperparameter (Marielle)
* Box Hyperparameter (Eugene)

---

## Team

| Member | Contributions |
|---|---|
| Martina Hokama | Project coordination: tasks, deadlines, review/correction of work, guideline allignment; Data Collection, Hyperparameter Tuning, Repository and Notebook review and correction, README, Presentation, Report writing |
| Marielle Reyes Lao | Data Collection, Hyperparameter Tuning, Presentation |
| Dalena Nguyen | Data Collection, Data Labelling & Preprocessing, Hyperparameter tuning, Presentation, and Editing |
| Eugene Yu | Data Collection, Data organization and Splitting, Baseline Model Training, Hyperparameter Tuning, Final Model Evaluation, Presentation, GitHub Repo Management, Meeting Notetaking, OneDrive Folder Management |


## Team Timeline

Oct 25  - Project Idea Finalization  - All Members 

Data Collection  - All Members (30 images/videos per member) 

Oct 28  - Data Labelling & Preprocessing  - Dalena Nguyen & Marielle Reyes Lao 

~Oct 30 - Model Training & Fine-Tuning - Eugene Yu 

~Nov 03 - Hyperparameter Tuning  - All Members 

~Nov 03  - Model Evaluation - Martina Hokama 

~Nov 06 - Final Fine-Tuning & Adjustments - All Members 

~Nov 06  - Report & Presentation Preparation - All Members 

Nov 08  - Presentation Video Editing  - Dalena Nguyen & Marielle Reyes Lao 

Nov 08  - Final Project Submission & Presentation -  All Members 



---

## Credits and licences

- **Roboflow "Cloud" dataset** by IP2, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): https://universe.roboflow.com/ip2-kbjz5/cloud-qthye
- **CCSN Database**, Zhang, J., Liu, P., Zhang, F., & Song, Q. (2018). *CloudNet: Ground-based cloud classification with deep convolutional neural network.* Geophysical Research Letters, 45, 8665–8672. CC0: https://github.com/upuil/CCSN-Database
- **Kaggle Cloud Types** by newgnart, MIT licence (notice included in the zip): https://www.kaggle.com/datasets/newgnart/cloud-types
- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **Label Studio** (HumanSignal): https://labelstud.io
- **WMO cloud classification**: https://wmo.int/world-meteorological-day-2017/classifying-clouds
