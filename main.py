#https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon


class ImageReader():
    """
    ImageReader is a simple abstraction on top of skimage and matplotlib that
    performs radon transforms and standard deivation analysis on png/jpeg files.

    Example usage:
      reader = ImageReader("path/to/image")
      reader.read(x1=0,y1=0,x2=100,y2=100)

    Users can also ROI manually using reader.set_subrect(...)
    And look at the corresponding radon transform and standard deviation columns
    independently with class methods radon_transform(...) and stddev_col(...)
    """

    
    def __init__(self, image_path):
        image_name_with_extension = os.path.basename(image_path)
        self.image_name = os.path.splitext(image_name_with_extension)[0]
        self._image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.x_max, self.y_max = self._image.shape

        #set subrect to whole image when initialized
        self.subrect = self._image
        self.subrect_pos = [[0, 0], [self.x_max, self.y_max]]
    

    def set_subrect(self, **kwargs):
        x1 = kwargs.get("x1", 0)
        y1 = kwargs.get("y1", 0)
        x2 = kwargs.get("x2", self.x_max)
        y2 = kwargs.get("y2", self.y_max)
        self.subrect = self._image[x1:x2, y1:y2]
        self.subrect_pos = [[x1, y1], [x2, y2]]


    def _radon_helper(self, steps=180):
        theta = np.linspace(0., 180., 180, endpoint=False)
        sinogram = radon(self.subrect, theta=theta)
        norm_sinogram = self._normalize(sinogram)
        return norm_sinogram


    def radon_transform(self, steps=180):
        norm_sinogram = self._radon_helper()
        [x1,y1], [x2,y2] = self.subrect_pos
        cv2.imwrite(f"images/{self.image_name}_radon_({x1},{y1})({x2},{y2}).jpg", norm_sinogram)
        cv2.imwrite(f"images/{self.image_name}_cropped_({x1},{y1})({x2},{y2}).jpg", self.subrect)
        cv2.imshow("cropped_image", self.subrect)
        cv2.imshow("radon_image", norm_sinogram)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
      

    def _normalize(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized = (image - min_val) / (max_val - min_val)  # Normalize to [0,1]
        return (normalized * 255).astype(np.uint8)
    

    def stddev_col(self):
        norm_sinogram = self._radon_helper()
        std = np.std(norm_sinogram, axis=0)
        plt.plot(std)
        plt.title("Standard Deviation of Columns in Image")
        plt.xlabel("Column")
        plt.ylabel("Standard Deviation")
        plt.show()

    
    def read(self, steps=180, **kwargs):
        x1, y1 = kwargs.get("x1", 0), kwargs.get("y1", 0)
        x2, y2 = kwargs.get("x2", self.x_max), kwargs.get("y2", self.y_max)
        self.set_subrect(x1=x1,y1=y1,x2=x2,y2=y2)
        self.subrect_pos = [[x1,y1], [x2,y2]]
        self.radon_transform(steps=steps)
        self.stddev_col(steps=steps)

  


ImageReader("images/AVG_20230130_M228_2L_zoom1pt5.jpg").read(x1=155, y1=130, x2=350, y2=340)
