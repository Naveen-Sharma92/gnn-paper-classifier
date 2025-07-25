# --- 1. Import Libraries ---
import streamlit as st
# The main library for building our web app.

import torch
# The main PyTorch library.

import torch.nn.functional as F
# Provides access to functions like activation functions.

from torch_geometric.datasets import Planetoid
# The loader for our Cora dataset.

from torch_geometric.nn import GCNConv
# The Graph Convolutional layer we used to build our model.


# --- 2. Re-define the GCN Model Architecture ---
# We need to define the exact same model structure as when we trained it,
# so we can load the saved weights into it correctly.

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# --- 3. Load the Trained Model and Data ---

# A function to load everything. Streamlit's cache decorator speeds this up.
@st.cache_resource
def load_model_and_data():
    # Load the Cora dataset (it will use the cached version)
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model with the correct dimensions
    model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
    
    # Load the saved weights into the model
    model.load_state_dict(torch.load('gcn_cora.pth', map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode
    model.eval()

    # Move data to the same device as the model
    data = data.to(device)

    return model, data, data.num_nodes

# Call the function to get our model and data
model, data, num_nodes = load_model_and_data()


# --- 4. Create the Streamlit User Interface ---

# Set the title of the web app
st.title("GNN Scientific Paper Classifier")

# Add a description
st.write(
    "Enter a paper ID (from 0 to", num_nodes - 1, ") to predict its subject "
    "using a Graph Neural Network trained on the Cora dataset."
)

# Create a number input field for the user
paper_id = st.number_input(
    f"Enter Paper ID (0-{num_nodes - 1})", 
    min_value=0, 
    max_value=num_nodes - 1, 
    value=0, 
    step=1
)

# Create a button to trigger the classification
if st.button("Classify Paper"):
    # Perform inference when the button is clicked
    with torch.no_grad(): # Disables gradient calculation for faster inference
        # The model needs the full graph to make a prediction
        out = model(data)
        
        # Get the prediction for the specific paper ID
        _, pred = out.max(dim=1)
        predicted_class = pred[paper_id].item()

        # Define the subject names for the Cora dataset
        class_names = [
            "Theory", "Reinforcement Learning", "Genetic Algorithms", 
            "Neural Networks", "Probabilistic Methods", "Case Based", "Rule Learning"
        ]

        # Display the result
        st.success(f"Paper {paper_id} is predicted to be in the subject: **{class_names[predicted_class]}**")

        # Get the true label for comparison
        true_class = data.y[paper_id].item()
        st.info(f"The actual subject of paper {paper_id} is: **{class_names[true_class]}**")