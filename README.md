# Smart Waste Sorter 

## Project Overview
- Smart Waste Sorter is an AI-based project designed to automatically classify waste items into categories such as Plastic, Paper, Metal, and Organic.
- This helps in promoting sustainability by encouraging proper waste segregation and recycling.

## Features
- Image classification using TensorFlow / Teachable Machine
- Classifies waste into recyclable and non-recyclable categories
- Can be integrated into a Flutter mobile app or web interface
- Lightweight model suitable for real-time predictions

## Tech Stack
- **Machine Learning:** TensorFlow / Keras / Teachable Machine
- **Programming Language:** Python
- **Frontend (Optional):** Flutter
- **Libraries:** NumPy, Pillow, TensorFlow Lite

## How It Works
- Upload or capture an image of waste.
- The trained model predicts the category (Plastic, Paper, Metal, Organic).
- Displays the classification result with confidence score.
- Optionally, suggests recycling instructions for recyclable items.

## Folder Structure
Week-1/
│
├── dataset/
│   ├── plastic/
│   ├── paper/
│   ├── metal/
│   └── organic/
│
├── train_model.py
├── waste_classifier.h5
├── test_image.jpg
└── README.md

## Impact
- This project contributes to sustainability by:
- Encouraging eco-friendly waste segregation
- Reducing landfill pollution
- Helping in smarter recycling management
