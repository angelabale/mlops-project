# mlops-project

## Project description

This project is part of the MLOps course final assessment.  
The goal is to design and implement an end-to-end machine learning pipeline following MLOps best practices, including reproducibility, version control, experimentation, and deployment.

At this stage (Checkpoint 1), the focus is on:
- Setting up a clean project structure  
- Managing the Python environment with UV  
- Implementing basic data loading and preprocessing  
- Running a first baseline training pipeline 

## Task definition

The objective of this project is to build a machine learning model that solves a supervised learning problem (classification or regression).

The model will:
- Load and preprocess the dataset  
- Train a baseline model  
- Evaluate performance using appropriate metrics  

The task and model complexity will remain simple initially, with the goal of emphasizing engineering quality and reproducibility rather than model performance.

The final task definition may evolve during the project.

## Data Source

The dataset used in this project comes from:  
**https://www.kaggle.com/datasets/nalisha/car-price-prediction-dataset/data**

Example:
- Public dataset from Kaggle / UCI / Open Data portal  
- Contains structured tabular data  
- Used for supervised learning  

A brief description of the dataset:
- Number of samples: TBD  
- Features: TBD  
- Target variable: TBD  

## Project structure 


```text
mlops-project/
├── src/            # Source code
    ├── data/ 
    ├── models/ 
├── tests/          # Unit tests
├── README.md
├── pyproject.toml
├── uv.lock
```

## How to Run

```bash
uv sync
```
