import streamlit as st
import torch
from PIL import Image
import numpy as np
import io
import pickle
from model_architecture import CNN
import base64
import matplotlib.pyplot as plt
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        self.remove_hooks()
        return cam


emoji_dict = {
    'Feu': '🔥',
    'Eau': '💧',
    'Air': '🌪️',
    'Vie': '🌱',
    'Magie': '✨',
    'Terre': '🪨',   
    'Tech': '⚙️'    
}

color_dict = {
    'Feu': '#ff9800',    
    'Eau': '#2196f3',    
    'Air': '#b3e5fc',    
    'Vie': '#4caf50',   
    'Magie': '#9c27b0',  
    'Terre': '#795548',  
    'Tech': '#cddc39'  
}


# Load model and encoder (update paths as needed)
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load('model_skylander.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_encoder():
    with open('encoder_skylander.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return encoder

model = load_model()
encoder = load_encoder()

st.title('Skylander Type Prediction')

uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])

predicted_label = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    width, height = image.size
    image_resized = image.resize((50, 50), resample=Image.LANCZOS)
    image_array = np.array(image_resized)
    image_tensor = torch.from_numpy(image_array).float() / 255.0
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_label = encoder.inverse_transform([predicted_class.item()])[0]
    emoji = emoji_dict.get(predicted_label, '')
    color = color_dict.get(predicted_label, '#cccccc')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    buf_resized = io.BytesIO()
    image_resized.save(buf_resized, format='PNG')
    img_resized_b64 = base64.b64encode(buf_resized.getvalue()).decode()

    width_resized, height_resized = image_resized.size

    gradcam = GradCAM(model, model.conv4)
    cam = gradcam(image_tensor, predicted_class.item())
    cam_resized = np.uint8(255 * cam)
    cam_resized = Image.fromarray(cam_resized).resize((width, height), resample=Image.BILINEAR)
    cam_colored = np.array(cam_resized.convert('RGB'))
    # Overlay heatmap on original image
    orig_img = np.array(image.convert('RGB'))
    heatmap = plt.get_cmap('jet')(cam / cam.max())[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((width, height), resample=Image.BILINEAR)
    overlay = np.uint8(0.5 * np.array(heatmap) + 0.5 * orig_img)
    # Display Grad-CAM result
    buf_cam = io.BytesIO()
    Image.fromarray(overlay).save(buf_cam, format='PNG')
    img_cam_b64 = base64.b64encode(buf_cam.getvalue()).decode()
    st.markdown(f"""
        <div style='display: flex; flex-direction: row; justify-content: center; gap: 32px; margin-bottom: 16px;'>
            <div>
                <img src='data:image/png;base64,{img_cam_b64}' style='border: 8px solid {color}; border-radius: 16px; box-shadow: 0 0 12px {color}; width: {width}px; height: {height}px; object-fit: contain;'>
                <div style='text-align: center; margin-top: 8px;'>Grad-CAM</div>
            </div>
            <div>
                <img src='data:image/png;base64,{img_b64}' style='border: 8px solid {color}; border-radius: 16px; box-shadow: 0 0 12px {color}; width: {width}px; height: {height}px; object-fit: contain;'>
                <div style='text-align: center; margin-top: 8px;'>Original</div>
            </div>
        </div>
        
        <div style='display: flex; justify-content: center;'>
            <span style='background: {color}; color: #fff; padding: 12px 32px; border-radius: 12px; font-size: 1.2em; font-weight: bold; box-shadow: 0 0 8px {color};'>
                Predicted Skylander Type: {predicted_label} {emoji}
            </span>
        </div>
        <div style='display: flex; justify-content: center; margin-top: 8px;'>
            <span style='background: #eee; color: #222; padding: 8px 24px; border-radius: 8px; font-size: 1em;'>
                <b>Above:</b> Grad-CAM heatmap shows what the AI focused on for its prediction.
            </span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info('Please upload an image to get a prediction.')

st.sidebar.title('About this App')
st.sidebar.markdown('''
**Skylander Type Prediction**

This app uses a Convolutional Neural Network (CNN) trained on images of Skylanders to predict the type (element) of a Skylander from a photo.

**How to use:**
- Upload a PNG or JPG image of a Skylander using the uploader on the main page.
- The AI will process the image and predict its type (Feu, Eau, Air, Vie, Magie).
- The border color and emoji will match the predicted type.

*Model: BetterCNN (PyTorch)*
''')
