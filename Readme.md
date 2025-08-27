# SoRCoF – Song Rating Collaborative Filtering

- **SoRCoF** is a prototype music recommender system built in **PyTorch**
- It extends earlier song rating–prediction work into a neural collaborative filtering model (user & song embeddings + MLP)
- The repo implements and validates the full pipeline (data preprocessing → model → training setup → inference)

---

## Features
- Neural collaborative filtering with **user & song embeddings + MLP**  
- Preprocessing pipeline using **LabelEncoder** and PyTorch `Dataset`/`DataLoader` batching  
- Training pipeline with **Adam optimizer** and learning rate scheduling  
- Demo notebook to showcase recommendations for example users and songs (inference)  
- Design intended to scale to millions of ratings (prototype runs were **single-epoch only**)

---

## Project structure
```
SoRCoF/
├── Notebooks/
│ ├── Project_RatingPredictions.ipynb → Main notebook: preprocessing, model, training setup
│ └── demo_notebook.ipynb → Demo: loads checkpoint & encoders, runs inference
│
├── Models/
│ ├── recsys_model_updated.pth → Saved model checkpoint
│ ├── lbl_user_updated.pkl → LabelEncoder object for users
│ └── lbl_song_updated.pkl → LabelEncoder object for songs
│
├── data/
│ └── songsDataset.csv → Dataset (included for convenience)
│
└── README.md → This file
```
---
## Pipeline (What's happening in the notebooks)

### A) In `Project_RatingPredictions.ipynb` 
### 1. Data Preprocessing
- Reads `songsDataset.csv` (`user_id`, `song_id`, `rating`)
- Uses `sklearn.preprocessing.LabelEncoder` to convert user/song IDs into integer indices
- Builds a custom PyTorch `Dataset` and `DataLoader` for batching

### 2. Model
- **2 embedding layers**: `user_embedding`, `song_embedding`
- Embeddings concatenated → passed through a **2-layer MLP** → outputs scalar rating prediction

### 3. Training Setup
- **Loss**: MSE (for rating regression)  
- **Optimizer**: Adam  
- **Learning rate scheduler**: present

### B) In `demo_notebook.ipynb`
- Loads a checkpoint file (e.g. `recsys_model_updated.pth`) and encoders (`lbl_user_updated.pkl`, `lbl_song_updated.pkl`) if available
- Accepts a `user_id` + candidate song list → encodes them → predicts scores → shows **Top-N recommendations**

---

## Evaluation

- **RMSE**: computed on validation set for rating prediction accuracy  
- **Precision@K / Recall@K**: computed for recommendation quality  

---

## Limitations 

- Not fully trained i.e., experiments were limited to single-epoch runs due to lack of consistent GPU access
- Demo recommendations may appear generic (same top songs) because embeddings have not converged
- This repo demonstrates a working prototype and the full pipeline, but not production-level trained weights

---

## Future Work

- Run **full multi-epoch training** on the dataset with GPU acceleration  
- **Hyperparameter tuning**: learning rate, embedding size, MLP depth, Regularization
- Investigate **cold-start strategies** (e.g., content features / hybrid models)

---
