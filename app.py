import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tempfile
import os

# Initialize session state for graph
if 'graph' not in st.session_state:
    st.session_state.graph = nx.DiGraph()

# Sidebar: Upload Dataset
st.sidebar.header("1️⃣ Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write("### Preview:")
    st.sidebar.dataframe(df.head())

    # Select Target Column
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    # Preprocessing Options
    st.sidebar.header("2️⃣ Select Preprocessing Steps")
    scale_data = st.sidebar.checkbox("Apply Standard Scaling")
    encode_labels = st.sidebar.checkbox("Encode Categorical Labels")

    # Model Selection
    st.sidebar.header("3️⃣ Select ML Model")
    model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest"])

    # Run Pipeline Button
    if st.sidebar.button("Run Pipeline"):
        # Preprocessing
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if encode_labels:
            y = LabelEncoder().fit_transform(y)
        if scale_data:
            X = StandardScaler().fit_transform(X)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate Model
        accuracy = accuracy_score(y_test, y_pred)
        st.sidebar.success(f"✅ Model Accuracy: {accuracy:.2f}")

        # Generate auto-code
        code = f'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("your_dataset.csv")
X = df.drop(columns=["{target_col}"])
y = df["{target_col}"]

{'y = LabelEncoder().fit_transform(y)' if encode_labels else ''}
{'X = StandardScaler().fit_transform(X)' if scale_data else ''}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {{accuracy:.2f}}")
        '''
        # Save and Download Python Script
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
            f.write(code.encode("utf-8"))
            st.sidebar.download_button("⬇ Download Auto-Generated Code", f.name, file_name="ml_pipeline.py")

# Main UI: Interactive Graph
st.title("🔗 Drag-and-Drop ML Pipeline Builder")
G = st.session_state.graph

# Ensure at least one node before rendering Pyvis graph
if G.number_of_nodes() > 0:
    net = Network(height="500px", width="100%", directed=True)

    # Add nodes and edges to Pyvis network
    for node in G.nodes:
        net.add_node(node, label=node, color="#3497db")
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])

    # Generate and display interactive graph
    temp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.show(temp_html.name)
    
    with open(temp_html.name, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=400)
else:
    st.warning("⚠ No pipeline components added yet. Drag & Drop components to build a pipeline.")
