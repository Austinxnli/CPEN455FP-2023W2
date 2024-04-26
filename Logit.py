import torch
import numpy as np
from torchvision import datasets, transforms
from dataset import *
from model import * 
from classification_evaluation import *
from generation_evaluation import *
from tqdm import tqdm
from utils import *


# Constants
NUM_CLASSES = 4  # Assuming you have 4 classes
TEST_SET_SIZE = 519  # The size of your test set

if __name__ == "__main__":
    # Optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--file_name', type=str,
                        default='test_logits', help='Result File Name.')
    parser.add_argument('-m', '--model_path', type=str,
                        default='models/conditional_pixelcnn.pth', help="Tained Model Path.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your model
    PATH = args.model_path
    model = PixelCNN(nr_resnet=1, nr_filters=80, input_channels=3, nr_logistic_mix=5)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Dataloader
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}
    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    test_dataset = CPEN455Dataset(root_dir='data', mode='test', transform=ds_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # Get logits
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    losses = np.zeros((TEST_SET_SIZE, NUM_CLASSES))
    
    print("Calculate Logits")
    with torch.no_grad():
        for i, (image, _) in enumerate(test_loader):
            if i%50==0: # Print the progress
                print(f"Image {i} Processing") 
            image = image.to(device)
            for label in range(NUM_CLASSES):
                label_tensor = torch.full((1,), label, dtype=torch.long).to(device)
                model_output = model(image, label_tensor)
                loss = discretized_mix_logistic_loss(image, model_output, training=False).item()
                losses[i, label] = loss

    # Ensure the logits are the correct shape
    assert losses.shape == (TEST_SET_SIZE, NUM_CLASSES), "Logits are not the correct shape"

    # Save the logits
    np.save(f'{args.file_name}.npy', losses)