# Project Title

## Overview

This project is aimed at segmenting aerial imagery for roof segmentation. The model is designed to handle image segmentation tasks using a U-Net architecture.

## How to Run the Project

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/recabet/AerialSegmentation
   cd AerialSegmentation
   ```
2. **Install the required dependencies**: You can install the necessary libraries using pip or conda. Here is a sample command to install the dependencies:
    ```bash
   pip install -r requirements.txt

    ```
3. **Download the dataset**: To download the latest dataset, run the following code:
    ```py
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("atilol/aerialimageryforroofsegmentation")
    
    print("Path to dataset files:", path)

    ```
4. **Run the main notebook**: Open and run the main.ipynb Jupyter notebook to train and test the model. You can run the notebook either in Jupyter Lab or Jupyter Notebook:

```bash
   jupyter notebook main.ipynb
```