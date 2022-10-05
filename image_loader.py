"""
This code is responsible for loading and returning images from a given directory.
"""
import cv2


class LoadImages:
    """ Loads the images from the specified directory. """

    def __init__(self, location, file_pattern, start_idx, stop_idx):
        """
        Initiates the variables and creates an ordered image list.
        Parameters
        ----------
        location (str): The file path to image dir
        file_pattern (str): '.jpg' or '.png'
        start_idx (int): the first number of the image sequence
        stop_idx  (int): the last number of the image sequence
        """
        self.images = []
        self.cnt = 0
        for i in range(start_idx, stop_idx + 1):
            file_name = file_pattern % i
            self.images.append(f'{location}/{file_name}')
        print(f'found {len(self.images)} images')

    def __len__(self):
        """ Returns the length of image list """
        return len(self.images)

    def next(self):
        """
        Sequentially loads an image from the directory.
        Returns
        ----------
        flag (bool): end of the list?
        image (ndarray): grayscale image
        """
        if self.cnt >= len(self.images):
            return False, None
        img = cv2.imread(self.images[self.cnt], cv2.IMREAD_GRAYSCALE)
        self.cnt += 1
        return True, img
