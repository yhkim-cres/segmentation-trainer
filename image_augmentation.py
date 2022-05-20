import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class ImgAug:
    def __init__(self, only_affine=False):
        self.only_affine = only_affine
        if self.only_affine:
            # sequential apply & random order
            self.aug = iaa.SomeOf((0, 6), [
                iaa.Fliplr(1),
                iaa.Rotate((-20, 20)),
                iaa.Affine(scale=(0.8, 1.2)),
                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
                iaa.ScaleX((0.8, 1.2)),
                iaa.ScaleY((0.8, 1.2))
            ], random_order=True)
        else:
            # sequential apply & random order
            self.aug = iaa.SomeOf((0, 8), [
                iaa.Fliplr(1),
                iaa.GammaContrast((0.7, 1.7)),
                iaa.Add((-30, 30)),
                iaa.Multiply((0.7, 1.2)),
                iaa.Sharpen(alpha=(0.0, 0.4)),
                iaa.Rotate((-20, 20)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
                iaa.GaussianBlur(sigma=(0, 2.0)),
                iaa.Affine(scale=(0.8, 1.2)),
                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
                iaa.ScaleX((0.8, 1.2)),
                iaa.ScaleY((0.8, 1.2))
            ], random_order=True)

    def apply_aug(self, img, mask):
        """
        img : HxWxC
        mask : HxW (각 픽셀은 정수값으로 레이블링)
        """
        segmap = SegmentationMapsOnImage(mask, shape=img.shape)
        aug_img, aug_mask = self.aug(image=img, segmentation_maps=segmap)

        return aug_img, aug_mask.get_arr()

# class ImgAug:
#     def __init__(self):
#         # augmentation tools
#         clahe = iaa.CLAHE(clip_limit=(2, 5))
#         sharpen = iaa.Sharpen(alpha=(0, 0.5))
#         flip_lr = iaa.Fliplr(0.5)
#         gau_noise = iaa.AdditiveGaussianNoise(scale=(0, 0.15*255))
#         #invert = iaa.Invert(0.5)
#         blur = iaa.GaussianBlur(sigma=(0.0, 3.0))
#         add = iaa.Add((-30, 30))
#         rotate = iaa.Rotate((-30, 30))

#         # sequential apply & random order
#         self.aug = iaa.SomeOf((0, 4), [
#             clahe,
#             sharpen,
#             flip_lr,
#             gau_noise,
#             blur,
#             add,
#             rotate
#         ], random_order=True)

#     def apply_aug(self, img, mask):
#         """
#         img : HxWxC
#         mask : HxW (각 픽셀은 정수값으로 레이블링)
#         """
#         segmap = SegmentationMapsOnImage(mask, shape=img.shape)
#         aug_img, aug_mask = self.aug(image=img, segmentation_maps=segmap)

#         return aug_img, aug_mask.get_arr()