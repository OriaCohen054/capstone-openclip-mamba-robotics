import torch
from PIL import Image
import open_clip

def test_openclip():
    print("Loading OpenCLIP model (this might take a minute the first time)...")
    
    # Load a lightweight model for testing purposes
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    
    # The exact path to the image from the previous script
    img_path = "data/pnp/2022-12-15_pnp_many_objects_in_env/2022-12-15_16-25-40/raw/traj_group0/traj13/images0/im_0.jpg"
    
    try:
        print(f"Loading image from: {img_path}")
        image = Image.open(img_path)
        
        # Preprocess the image (resize and normalize) and add batch dimension
        image_input = preprocess(image).unsqueeze(0)
        
        print(f"Image preprocessed. Tensor shape: {image_input.shape}")
        
        # Extract features (Embeddings) using OpenCLIP
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        print("Success: Image passed through OpenCLIP.")
        print(f"Final Embedding shape: {image_features.shape}")
        
    except FileNotFoundError:
        print("Error: Could not find the image. Please verify the" \
        " path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_openclip()