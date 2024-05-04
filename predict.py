import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

class ImageClassifier:
    def __init__(self):
        self.parse_args()
        self.load_checkpoint()
        self.load_category_names()
        self.device = torch.device('cuda' if self.args.gpu and torch.cuda.is_available() else 'cpu')

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('image_path', type=str, help='Path to the image file')
        parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
        parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
        parser.add_argument('--category_names', help='Path to category names mapping JSON file')
        parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
        self.args = parser.parse_args()

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.args.checkpoint_path)
            self.model = checkpoint['model']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.class_to_idx = checkpoint['class_to_idx']
        except FileNotFoundError:
            print(f"Checkpoint file not found: {self.args.checkpoint_path}")
            exit(1)

    def load_category_names(self):
        if self.args.category_names is None:
            self.cat_to_name = None
        else:
            with open(self.args.category_names, 'r') as f:
                self.cat_to_name = json.load(f)

    def process_image(self, image_path):
        '''
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        '''
        # Open the image
        with Image.open(image_path) as img:
            # Resize the image where the shortest side is 256 pixels
            width, height = img.size
            aspect_ratio = width / height
            new_size = (256, int(256 / aspect_ratio)) if width < height else (int(256 * aspect_ratio), 256)
            img.thumbnail(new_size)

            # Crop out the center 224x224 portion of the image
            left = (new_size[0] - 224) / 2
            top = (new_size[1] - 224) / 2
            right = left + 224
            bottom = top + 224
            img = img.crop((left, top, right, bottom))

            # Convert color channels to 0-1
            np_image = np.array(img) / 255.0

            # Normalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            np_image = (np_image - mean) / std

            # Reorder dimensions for PyTorch
            np_image = np_image.transpose((2, 0, 1))

            return np_image


    def imshow(image, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes it is the third dimension
        image = np.array(image)
        image = image.transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        # Convert image to float32
        image = image.astype(np.float32)

        ax.imshow(image)
        ax.axis('on')

        return ax

    def predict(self):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        topk = self.args.top_k
        image_path = self.args.image_path
        model = self.model

        # Process the image
        img = self.process_image(image_path)

        # Convert the numpy array to a tensor and normalize
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)

        # Set the model to evaluation mode
        model.eval()

        # Disable gradient calculation to speed up the inference
        with torch.no_grad():
            # Forward pass
            output = model(img_tensor)

        # Calculate the class probabilities
        probabilities = torch.exp(output)

        # Get the top k probabilities and classes
        top_probabilities, top_classes = probabilities.topk(topk)

        # Convert indices to class labels
        idx_to_class = {idx: class_ for class_, idx in sorted(model.class_to_idx.items(), key=lambda x: x[1])}
        top_classes = [idx_to_class[idx.item()] for idx in top_classes[0]]

        if self.args.category_names:
            # If category_names is specified, map the index to class names using cat_to_name dictionary
            class_names = [self.cat_to_name[class_idx] for class_idx in top_classes]
        else:
            # If category_names is not specified, use the index as the class name
            class_names = top_classes

        return top_probabilities[0].tolist(), class_names


if __name__ == "__main__":
    classifier = ImageClassifier()
    probs, classes = classifier.predict()

    for i in range(len(probs)):
        print(f"{classes[i]}: {probs[i]*100:.2f}%")

