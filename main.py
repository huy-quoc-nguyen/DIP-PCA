import cv2
import glob
import numpy as np
import settings

from lazylog import LazyLog
from pca import PCA


class PCAGrayScale(LazyLog):
    def __init__(self, img_dataset: np.ndarray):
        """
        :param img_dataset: An image dataset, which is a matrix with the shape of (N x H x W), where:
                - N: number of images
                - H: height of images
                - W: width of images
                - each item of the matrix is an real value between 0 and 1
            Notes: All images should have same width and height
        """
        # Get the shape of the input data
        super().__init__()
        assert len(img_dataset.shape) == 3
        self._n_samples, self._height, self._width = img_dataset.shape
        self.logger.info({
            'msg': 'Image dataset shape',
            'shape': img_dataset.shape
        })

        self._n_features = self._height * self._width

        # Flatten the images of shape (height, width) to vectors of length height x width
        self._flatten_dataset = img_dataset.reshape((self._n_samples, self._n_features))

        # Build the PCA transformer
        self._pca_transformer = PCA(self._flatten_dataset)

    def transform(self, img_dataset: np.ndarray, n_reduced_features: int) -> np.ndarray:
        # Get the shape of the input data
        assert len(img_dataset.shape) == 3
        n_samples, height, width = img_dataset.shape
        n_features = height * width

        # Flatten the images of shape (height, width) to vectors of length height x width
        flatten_dataset = img_dataset.reshape((n_samples, n_features))
        projected_dataset = self._pca_transformer.project_and_restore(flatten_dataset, n_reduced_features)
        projected_dataset = np.array(np.round(projected_dataset), dtype=np.uint8)

        return projected_dataset.reshape((n_samples, height, width))


def read_data():
    img_dataset = None
    for image_path in glob.glob(settings.DATA_PATH):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        width //= 2
        height //= 2
        img = cv2.resize(img, (width, height))
        img = img.reshape((1, height, width))
        if img_dataset is None:
            img_dataset = img
        else:
            img_dataset = np.vstack((img_dataset, img))
    return img_dataset


def main():
    logger = LazyLog()
    img_dataset = read_data()
    pca_gray = PCAGrayScale(img_dataset)
    projected_image_dataset = pca_gray.transform(img_dataset, 16)
    for i in range(projected_image_dataset.shape[0]):
        logger.logger.debug({
            'msg': 'Info of {}-th image'.format(i),
            'originalShape': img_dataset[i].shape,
            'projectedShape': projected_image_dataset[i].shape
        })
        cv2.imwrite('./resources/src_{}.pgm'.format(i), img_dataset[i])
        cv2.imwrite('./resources/dst_{}.pgm'.format(i), projected_image_dataset[i])


if __name__ == '__main__':
    main()
