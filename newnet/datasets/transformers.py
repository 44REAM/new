# pylint: disable=redefined-outer-name

import numpy as np


from albumentations import (
    HorizontalFlip,
    ElasticTransform,
    Rotate,
    IAAAffine,
    ShiftScaleRotate
)



class Transforms():
    def __init__(self, basic=True, elastic_transform=False,
                 shift_scale_rotate=False):
        transforms = []

        if basic:
            transforms.append(HorizontalFlip(p=0.5))
            transforms.append(Rotate(p=0.8,  limit=10))

        if elastic_transform:
            transforms.append(ElasticTransform(p=0.2))

        if shift_scale_rotate:
            transforms.append(ShiftScaleRotate(p=0.2))

        #transforms.append(IAAAffine(p=1, shear=0.2, mode="constant"))
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image=image)['image']
        return image


if __name__ == '__main__':

    from urllib.request import urlopen
    import matplotlib.pyplot as plt
    import cv2

    def download_image(url):
        data = urlopen(url).read()
        data = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    image = download_image(
        'https://d177hi9zlsijyy.cloudfront.net/wp-content/uploads/sites/2/2018/05/11202041/180511105900-atlas-boston-dynamics-robot-running-super-tease.jpg')

    print(image.shape)
    transform = Transforms(basic=True)
    img = transform(image)
    plt.imshow(img)
    plt.show()
