import torch
import pandas as pd
from model import * 
from dataset import *
from classification_evaluation import *
from torchvision import transforms
from tqdm import tqdm
from pprint import pprint

if __name__ == '__main__':
    # Optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str,
                        default='hugging_face_result', help='Tag of the result file.')
    parser.add_argument('-m', '--model_path', type=str,
                        default='models/conditional_pixelcnn.pth', help="Path to load trained model.")
    parser.add_argument('-f', '--fid_score', type=float,
                        default=-1.0, help="FID score to add at the end of the file if any.")
    
    args = parser.parse_args()
    pprint(args.__dict__)

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = PixelCNN(nr_resnet=1, nr_filters=80, input_channels=3, nr_logistic_mix=5)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()

    # Build dataloader
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}
    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    test_dataset = CPEN455Dataset(root_dir='data', mode='test', transform=ds_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)
    print("Dataset size:", len(test_dataset))       
    print("DataLoader size:", len(test_loader))

    # Build Datafram
    classification = pd.read_csv('data/test.csv', names=['id', 'label'])
    classification['id'] = classification['id'].str.replace('test/', '')

    # Get labels
    labels = []
    for batch_idx, item in enumerate(tqdm(test_loader)):
        model_input, _ = item
        labels.append(get_label(model, model_input.to(device), device))

    classification['label'] = torch.cat(labels).cpu().numpy()
    pprint(classification)

    # Add FID score at the end if provided
    if args.fid_score > 0:
        classification.loc[len(classification)] = ["FID", args.fid_score]

    # Save as CSV file
    classification.to_csv(f"test_results/{args.tag}.csv", index=False)