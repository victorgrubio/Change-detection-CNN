import cv2
import numpy as np
cimport numpy as np
import sys


"""
This class contains methods used in multiple scripts in order to compact
the code and above repetition related to img processing
"""
class Aligner():
    def __init__(self):
        self.features = 500  # max number of features
        self.matches_ratio = 0.15  # portion of features finally obtained
        self.orb = cv2.ORB_create(self.features)
        self.matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)


def get_img_dims(img, max_width, max_height, model_input):
    """
    Resize img according to model input and max dimensions.
    """
    width = img.shape[1]
    height = img.shape[0]
    if width > max_width:
        width = max_width
    elif width % model_input != 0:
        width = (int(width/model_input) + 1)*model_input
    if height > max_height:
        height = max_height
    elif height % model_input != 0:
        height = (int(height/model_input) + 1)*model_input
    return width, height

def cut_img(img, img_cuts_dict):
    """
    Cut a determined portion of each img's side.
    """
    if img_cuts_dict['cut_bot'] == 0:
        img_cuts_dict['cut_bot'] = img.shape[0]
    if img_cuts_dict['cut_right'] == 0:
        img_cuts_dict['cut_right'] = img.shape[1]
    final_img = img[
        img_cuts_dict['cut_top']:img_cuts_dict['cut_bot'],
        img_cuts_dict['cut_left']:img_cuts_dict['cut_right']]
    return final_img


def align_imgs(aligner, img1, img2):
    # A partir de aqui se aplica cada imagen
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    points1, descriptors1 = aligner.orb.detectAndCompute(img1Gray, None)
    points2, descriptors2 = aligner.orb.detectAndCompute(img2Gray, None)
    # Match features from both imgs
    matches = aligner.matcher.match(descriptors1, descriptors2, None)
    # Sort matches
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Get only the best matches
    cdef int numGoodMatches = int(len(matches) * aligner.matches_ratio)
    matches = matches[:numGoodMatches]
    # Draw matches on both imgs
    imMatches = cv2.drawMatches(img1, points1, img2, points2, matches, None)
    # Final points locations
    cdef np.ndarray final_points1 = np.zeros(
        (len(matches), 2), dtype=np.float32)
    cdef np.ndarray final_points2 = np.zeros(
        (len(matches), 2), dtype=np.float32)
    cdef int index
    for index in range(len(matches)):
        # Get point coordinates
        final_points1[index, :] = points1[matches[index].queryIdx].pt
        final_points2[index, :] = points2[matches[index].trainIdx].pt
    # Find homography matrix
    h, mask = cv2.findHomography(final_points1, final_points2, cv2.RANSAC)
    cdef int height, width, channels
    height, width, channels = img2.shape
    img1Reg = cv2.warpPerspective(img1, h, (width, height))
    return img1Reg, h


def morph_erode(img, kernel_size):
    """
    applies morphological dilate transformation to an img (normally a mask)
    """
    cdef np.ndarray kernel = np.ones(
        (kernel_size, kernel_size), dtype=np.uint8)
    cdef np.ndarray dilation = cv2.morphologyEx(
        img, cv2.MORPH_ERODE, kernel)
    return dilation

def morph_dilate(img, kernel_size):
    """
    applies morphological dilate transformation to an img (normally a mask)
    """
    cdef np.ndarray kernel = np.ones(
        (kernel_size, kernel_size), dtype=np.uint8)
    cdef np.ndarray dilation = cv2.dilate(
        img, kernel, iterations=1)
    return dilation


def morph_closing(img, kernel_size):
    """
    applies morphological closing transformation to an img (normally a mask)
    """
    cdef np.ndarray kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cdef np.ndarray closing_img = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, kernel)
    return closing_img

def morph_opening(img, kernel_size):
    """
    applies morphological closing transformation to an img (normally a mask)
    """
    cdef np.ndarray kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cdef np.ndarray opening_img = cv2.morphologyEx(
        img, cv2.MORPH_OPEN, kernel)
    return opening_img

def find_contours(mask_img):
    """
    Find contours
    """
    cdef np.ndarray img_tmp
    img_tmp, cnts, hierarchy = cv2.findContours(
        mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # keep only the largest ones
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    return cnts


def draw_contours(img, cnts):
    """
    Draw bounding boxes for the specified contours in an img
    if they are big enough
    """
    cdef int img_area = img.shape[0] * img.shape[1]
    cdef float threshold_min = 0.001
    cdef float threshold_max = 0.5
    cdef int index
    cdef int x, y, w, h
    cdef float contour_relative_area
    cdef int box_size = min([int(max(img.shape) / 320), 1])
    for index in range(len(cnts)):
        contour_relative_area = (cv2.contourArea(cnts[index]) * 10) / img_area
        if contour_relative_area < threshold_max and\
                contour_relative_area > threshold_min:
            x, y, w, h = cv2.boundingRect(cnts[index])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), box_size)
    return img


def create_video(video_name, frame_rate, video_shape, codec='MJPG'):
    """
    Create video file to write frames on it
    To write a frame use: video.write(frame)
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, float(frame_rate), video_shape)
    return video


def create_windows(img_dict):
    """
    create an specified number of windows for img plotting
    """
    for name, img in list(img_dict.items()):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)


def show_imgs(img_dict, auto=False):
    """
    shows img and tries to update them (not implemented yet)
    """
    k = 0
    for name, img in list(img_dict.items()):
        if img is not None:
            cv2.imshow(name, img)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
