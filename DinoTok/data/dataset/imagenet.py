import os
import math
import random
import numpy as np
from typing import Optional, Tuple
from PIL import Image
from torchvision.datasets import ImageNet
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def random_crop_arr(pil_image, image_size, min_crop_frac=0.88, max_crop_frac=1.0):
    """
    Args:
        pil_image (PIL.Image): 输入图像
        image_size (int or tuple): 目标裁剪大小 (h, w) 或 单个 int (h=w)
        min_crop_frac (float): 最小随机裁剪比例
        max_crop_frac (float): 最大随机裁剪比例
    Returns:
        PIL.Image
    """

    # 兼容 image_size 是 int 或 (h, w)
    if isinstance(image_size, int):
        out_h = out_w = image_size
    else:
        out_h, out_w = image_size

    min_smaller_dim_size = math.ceil(min(out_h, out_w) / max_crop_frac)
    max_smaller_dim_size = math.ceil(max(out_h, out_w) / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # 下采样（保持质量）
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # 按比例缩放，确保短边 >= smaller_dim_size
    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # 转 numpy 裁剪
    arr = np.array(pil_image)
    if arr.shape[0] < out_h or arr.shape[1] < out_w:
        raise ValueError(f"目标裁剪尺寸 {image_size} 大于当前图像大小 {arr.shape[:2]}")

    crop_y = random.randrange(arr.shape[0] - out_h + 1)
    crop_x = random.randrange(arr.shape[1] - out_w + 1)
    return Image.fromarray(arr[crop_y : crop_y + out_h, crop_x : crop_x + out_w])


class ImageNet(VisionDataset):
    num_classes = 1000

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=default_loader,
        split="train",
        max_retries: int = 10,
        keep_badlist: bool = True,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.loader = loader
        self.split = split
        self.max_retries = max_retries
        self.keep_badlist = keep_badlist

        self.samples = self.make_dataset()
        self._bad_indices = set()

    def make_dataset(self):
        instances = []
        split_dir = os.path.join(self.root, self.split)
        class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(split_dir)))
        }
        for cls_name, idx in class_to_idx.items():
            cls_folder = os.path.join(split_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                instances.append((os.path.join(cls_folder, img_name), idx))
        random.shuffle(instances)
        print(
            f"Split: {self.split}, number of samples: {len(instances)}, number of classes: {len(class_to_idx)}"
        )
        return instances

    def _try_load(self, index: int) -> Tuple[Image.Image, int]:
        """单次尝试读取并做 transform，成功则返回，否则抛异常"""
        path, target = self.samples[index]
        img = self.loader(path)  # 可能在这里抛错
        if self.transform is not None:
            img = self.transform(img)  # 也可能在这里抛错（如随机裁剪越界）
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __getitem__(self, index: int):
        # 如果这个 index 已经被标记为坏样本，直接换一个
        if index in self._bad_indices:
            index = self._sample_new_index()

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._try_load(index)
            except Exception as e:
                last_err = e
                if self.keep_badlist:
                    # 标记为坏样本，后续避开
                    self._bad_indices.add(index)
                # 随机换一个索引再试（避免卡在同一张坏图）
                index = self._sample_new_index()

        # 多次重试仍失败，抛出带路径信息的错误更易定位
        path, _ = self.samples[index]
        raise RuntimeError(
            f"[ImageNet {self.split}] Failed to load after {self.max_retries} retries. "
            f"Last error on '{path}': {repr(last_err)}"
        )

    def _sample_new_index(self) -> int:
        """采样一个未在坏名单中的随机索引；若坏图太多，仍可能采到坏的，但会被重试机制兜底"""
        n = len(self.samples)
        for _ in range(20):  # 尝试若干次尽量避开坏索引
            j = random.randint(0, n - 1)
            if j not in self._bad_indices:
                return j
        # 退化：坏索引太多时直接返回随机索引
        return random.randint(0, n - 1)

    def __len__(self) -> int:
        return len(self.samples)


def build_imagenet(
    data_path="/high_perf_store2/users/xiexuezhen/Imagenet/imagenet",
    final_reso=512,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
):
    train_aug = [
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, final_reso)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std, inplace=True),
    ]
    train_aug = transforms.Compose(train_aug)

    val_aug = [
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, final_reso)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std, inplace=True),
    ]

    val_aug = transforms.Compose(val_aug)
    return ImageNet(data_path, transform=train_aug, split="train"), ImageNet(
        data_path, transform=val_aug, split="val"
    )
