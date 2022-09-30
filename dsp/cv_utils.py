import cv2

def to_grayscale(img):
  if len(img.shape) == 2: return img
  if img.shape[2] == 1:
    return img.reshape(img.shape[:2])
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

