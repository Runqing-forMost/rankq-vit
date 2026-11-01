from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, Dataset
from glob import glob
from tqdm import tqdm

dict_ = {
    'oxford_pets': {'mean': (0.4810887, 0.44917485, 0.39570457), 
                    'std': (0.2644624, 0.2596167, 0.26807153)},
    'oxford_flowers': {'mean':(0.43553895, 0.37773782, 0.28795487),
                       'std': (0.29411197, 0.24257936, 0.2674491)},
    'imagenet-o': {
        'mean':(0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }
}

transform_dict = {
    'oxford_pets': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(dict_['oxford_pets']['mean'], dict_['oxford_pets']['std'])
    ]),
    'oxford_flowers': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(dict_['oxford_flowers']['mean'], dict_['oxford_flowers']['std'])
    ]),
    'imagenet-o': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(dict_['imagenet-o']['mean'], dict_['imagenet-o']['std'])
])
}

class BasicDatasets(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        image_paths = []
        for ext in ['jpg']:
            image_paths.extend(glob(os.path.join(data_dir, '**', f'*{ext}'), recursive=True))
            image_paths.extend(glob(os.path.join(data_dir, '**', f'*{ext.upper()}'), recursive=True))
        
        image_paths = sorted(set(image_paths))
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            image =  img.convert('RGB')
            pass
            
        if self.transform:
            image = self.transform(image)
        return image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


class ImageNetOOD(Dataset):
    """
    ImageNet-O Dataset for OOD calibration / sensitivity analysis.
    Labels are ignored by default.
    """

    def __init__(
        self,
        root,
        transform=None,
        max_samples=None,
        return_path=False,
    ):
        """
        Args:
            root (str): path to ImageNet-O root directory
            transform (callable): torchvision transform
            max_samples (int, optional): limit number of samples (for fast OOD probing)
            return_path (bool): whether to return image path
        """
        self.root = root
        self.transform = transform
        self.return_path = return_path

        self.samples = self._gather_images(root)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"[ImageNet-OOD] Loaded {len(self.samples)} images")

    def _gather_images(self, root):
        images = []
        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if is_image_file(fname):
                    images.append(os.path.join(class_path, fname))
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img, path
        else:
            return img
        

def compute_mean_std_dataset(
    data_dir:dir
):
    image_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor() 
    ])
    dataset = BasicDatasets(data_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    print('computing mean and std ...')
    mean, mean_sq = torch.zeros(3).cuda(), torch.zeros(3).cuda()
    total_pixels = 0
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch = batch.cuda()
        batch_size, channels, height, width = batch.shape
        mean += batch.sum(dim=(0, 2, 3))
        mean_sq += (batch ** 2).sum(dim=(0, 2, 3))
        total_pixels += batch_size * height * width
    
    mean /= total_pixels
    std = torch.sqrt(mean_sq / total_pixels - mean ** 2)
    mean = mean.cpu()
    std = std.cpu()
    print(f"Mean: {mean.numpy()}")
    print(f"Std: {std.numpy()}")
    return mean, std

if __name__ == '__main__': # to do test
    # compute_mean_std_dataset('/media/jrq/vlm-tta/oxford_pets/images')
    compute_mean_std_dataset('/media/jrq/vlm-tta/oxford_flowers/102flowers/jpg')
    