import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np


class PlantVillageClassifierStreamlit:
    def __init__(self, model_path, class_labels_path="class_label.txt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_labels = self.load_class_labels(class_labels_path)
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()

    def load_class_labels(self, path):
        try:
            with open(path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            st.warning("Class label file not found. Using placeholder labels.")
            return [f"Class_{i}" for i in range(38)]

    def load_model(self, model_path):
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.class_labels))
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            top_class = np.argmax(probabilities)
            confidence = probabilities[top_class]
            return self.class_labels[top_class], confidence


class StreamlitApp:
    def __init__(self, model_path, class_labels_path="class_labels.txt"):
        self.classifier = PlantVillageClassifierStreamlit(model_path, class_labels_path)

    def run(self):
        st.set_page_config(page_title="Leaf Classifier", layout="centered")
        st.title("Dataset Leaf Classifier")
        st.markdown("Upload a plant leaf image and get its predicted disease or condition.")

        uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")

            with st.spinner("Analyzing the image..."):
                pred_class, confidence = self.classifier.predict(image)

            st.success(f"**Predicted Class:** {pred_class}")
            st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")


if __name__ == "__main__":
    app = StreamlitApp("plant_resnet50.pth", "class_label.txt")
    app.run()
