import os
import argparse
import warnings
import utils
import model
from grad_cam import GradCAM
from plot_utils import apply_mask


def get_args():
    parser = argparse.ArgumentParser(
        prog="GradCAM on Chest X-Rays",
        description="Overlays given label's CAM on a given Chest X-Ray."
    )
    parser.add_argument(
        '--image_path', type=str, default='./assets/original.jpg',
        help='Path to chest X-Ray image.'
    )
    parser.add_argument(
        '--label', type=str, default=None,
        choices=[
            'atelectasis',
            'cardiomegaly',
            'consolidation',
            'edema',
            'enlarged_cardiomediastinum',
            'lung_lesion',
            'lung_opacity',
            'normal',
            'pleural_effusion',
            'pneumonia',
            'pneumothorax'
        ],
        help='Choose from covid_19, lung_opacity, normal & pneumonia,\n'
        'to get the corresponding CAM.\n'
        'If not mentioned, the highest scoring label is considered.'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['vgg16', 'resnet18', 'densenet121'],
        help='Choose from vgg16, resnet18 or densenet121.'
    )
    parser.add_argument(
        '--output_path', type=str, default='./outputs/output.jpg',
        help='Format: "<path> + <file_name> + .jpg"'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # get path of the pretrained model checkpoint
    path = {
        # 'vgg16': './models/lr3e-5_vgg16_cuda.pth',
        'resnet18': '/kaggle/input/medical-resnet152/best_medical_resnet152.pth',
        # 'densenet121': './models/lr3e-5_densenet_cuda.pth'
    }
    path = path[args.model]

    if not os.path.exists(path):
        raise Exception(
            f'{path} not found.\n'
            'Download the required model from the following link.\n'
            'https://drive.google.com/drive/folders/'
            '14L8wd-d2a3lvgqQtwV-y53Gsnn6Ud2-w'
        )

    # load the model using pretrained weights
    model = eval(
        f'networks.get_{args.model}(out_features=4, path="{path}")'
    ).cpu()

    # set target layer for CAM
    if args.model == 'vgg16' or args.model == 'densenet121':
        target_layer = model.features[-1]
    elif args.model == 'resnet18':
        target_layer = model.layer4[-1]

    # get given label's index
    label = {
        'atelectasis': 0,
        'cardiomegaly': 1,
        'consolidation': 2,
        'edema': 3,
        'enlarged_cardiomediastinum': 4,
        'lung_lesion': 5,
        'lung_opacity': 6,
        'normal': 7,
        'pleural_effusion': 8,
        'pneumonia': 9,
        'pneumothorax': 10
    }

    idx_to_label = {v: k for k, v in label.items()}
    if args.label is not None:
        label = label[args.label]
    else:
        label = None

    # load and preprocess image
    image = utils.load_image(args.image_path)

    warnings.filterwarnings("ignore", category=UserWarning)
    # pass image through model and get CAM for the given label
    cam = GradCAM(model=model, target_layer=target_layer)
    label, mask = cam(image, label)
    print(f'GradCAM generated for label "{idx_to_label[label]}".')

    # deprocess image and overlay CAM
    image = utils.deprocess_image(image)
    image = apply_mask(image, mask)

    # save the image
    utils.save_image(image, args.output_path)