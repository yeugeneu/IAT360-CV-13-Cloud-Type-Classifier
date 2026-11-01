# Cloud Type Detection ☁️
A deep learning-based computer vision project for multi-class cloud object detection and short-term weather prediction (nowcasting). Developed by CV Group 13.

## 📋 Project Overview
Accurate cloud detection plays a vital role in short-term weather forecasting, climate studies, and outdoor event planning. This project implements a supervised deep learning model capable of classifying various cloud types—and supporting multi-label classification when multiple cloud formations appear simultaneously.

## Key Use Cases
If you see a weird cloud shape and are curious about it, it will tell you what cloud you have + the relation to the weather

Why would this be useful? Have you ever looked up at a strange cloud and wondered what it was? Most people can’t tell a cumulus from a stratocumulus, and cloud type is one of the simplest visible clues to what the weather will do next. A cirrus sky often comes before a front, while cumulonimbus means storms. The model could become an app that tells you what cloud you’re looking at and what it usually means for the weather, which would also be useful for curious people, students, hikers, and anyone deciding whether to bring rain gear.

## Possible extension:
* Climate Research: Studying how cloud formations impact Earth's temperature and climate patterns.
* Safety & Operations: Assisting with storm tracking and weather-dependent activities.

## ☁️ Cloud Labels & Dataset
The project utilizes the Cloud Classification Dataset featuring categories such as:

1. Altocumulus (AC)
2. Altostratus (AS)
3. Cirrocumulus (CC)
4. Cirrostratus (CS)
5. Cirrus (CI)
6. Contrails (CT) - While classified as 'Cirrus homogenitus' under official WMO guidelines, keeping them distinct optimizes object detection due to their unique linear geometry and distinct climate research value.
7. Cumulonimbus (CB)
8. Cumulus (CU)
9. Nimbostratus (NS)
10. Stratocumulus (SC)
11. Stratus (ST) 

We will be aiming for around 200 images for each class in addition to an extra 30 original photos for each group member.

## 👥 Team & Task Distribution
Eugene Yu: GitHub & Jupyter Notebook repository management, dataset storage organization (OneDrive/Google Drive), model training & fine-tuning.

Dalena Nguyen & Marielle Reyes Lao: Data collection, labelling, preprocessing, and presentation video editing.

Martina Hokama: Model evaluation and further fine-tuning.

All Members: Project idea finalization, data collection, hyperparameter tuning, and report/presentation preparation.

## 📅 Project Timeline & Milestones
* Oct 25: Project Idea Due
* ~Oct 28: Data Collection, Labelling & Preprocessing
* ~Nov 03: Baseline Model Training & Fine-Tuning
* ~Nov 06: Model Evaluation & Final Hyperparameter Tuning\
* Nov 08: Final Project Submission & Presentation
