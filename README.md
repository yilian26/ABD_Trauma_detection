# ABD_Trauma_detection

A deep learning pipeline for multi-organ segmentation and injury classification from abdominal trauma CT scans.

## Features

- Multi-organ segmentation using UNet and TotalSegmentator
- Injury grading based on ResNet + UNet hybrid architecture
- Configurable via `.ini` files
- Training logs and visualizations included

## Folder Structure
ABD_Trauma_detection/
├── config/
├── data/
├── models/
├── utils/
├── weights/ # (excluded via .gitignore)
├── log/
├── multiorgan_seg_train.py
├── README.md

## Requirements

- Python 3.8+
- PyTorch
- MONAI
- NumPy, matplotlib, etc.

## How to Run

```bash
python multiorgan_seg_train.py --config config/multiple/your_config.ini
