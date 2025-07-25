GNN Scientific Paper Classifier 🧠
This is an end-to-end machine learning project that uses a Graph Neural Network (GNN) to classify scientific papers into one of seven subjects based on their citation network. The project is built with PyTorch Geometric and deployed as an interactive web application using Streamlit.

Live Demo
You can try the live application here:
https://huggingface.co/spaces/Creator-1/GNN-Paper-Classifier

Features
Graph Neural Network: Implements a Graph Convolutional Network (GCN) to learn from the relationships (citations) between papers, achieving high accuracy on the Cora dataset.

Interactive UI: A simple and user-friendly web interface built with Streamlit that allows for real-time predictions.

Lightweight & Deployable: The trained model is very small (~17 KB), allowing for easy and free deployment on cloud platforms like Hugging Face Spaces. This project successfully addresses the deployment challenges faced with larger models.

Tech Stack
Model: Python, PyTorch, PyTorch Geometric

Web App: Streamlit

Dataset: Cora

Setup and Usage
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/YourUsername/gnn-paper-classifier.git

Create and activate a conda environment:

conda create --name gnn_project python=3.10 -y
conda activate gnn_project

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py
