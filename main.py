#https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html

import os
import cv2
import numpy as np
from skimage.transform import radon


class ImageReader():
  def __init__(self, image_path):
    image_name_with_extension = os.path.basename(image_path)
    self.image_name = os.path.splitext(image_name_with_extension)[0]
    self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    self.x_max, self.y_max = self.image.shape
  
  def get_subrect(self, **kwargs):
    x1 = kwargs.get("x1", 0)
    y1 = kwargs.get("y1", 0)
    x2 = kwargs.get("x2", self.x_max)
    y2 = kwargs.get("y2", self.y_max)
    
    radon_trans_image = self._normalize(self._radon_transform(self.image[x1:x2, y1:y2])) #Radon transform of cropped image
    cv2.imwrite(f"images/{self.image_name}_radon_({x1},{y1})({x2},{y2}).jpg", radon_trans_image)
    cv2.imwrite(f"images/{self.image_name}_cropped_({x1},{y1})({x2},{y2}).jpg", self.image[x1:x2, y1:y2])
    cv2.imshow("cropped_image", self.image[x1:x2, y1:y2])
    cv2.imshow("radon_image", radon_trans_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  def _radon_transform(self, image, steps=180):
    theta = np.linspace(0., 180., steps, endpoint=False)
    sinogram = radon(image, theta=theta)
    return sinogram

  def _normalize(self, image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)  # Normalize to [0,1]
    return (normalized * 255).astype(np.uint8)
  




ImageReader("AVG_20230406_M228N_wk8_2L_zoom1pt5.jpg").get_subrect(x1=155, y1=130, x2=350, y2=340)

