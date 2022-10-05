"""
This code is responsible for predicting the car's trajectory using visual odometry
based on image features and epipolar geometry.
"""
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_loader import LoadImages


matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
plt.rc('legend', **{'fontsize': 12})


class VisualOdometry:
    """
    * Visual Odometry Class *
    It contains methods specific to visual odometry that primarly extract and match image features
    and then estimate the relative image pose using established image correspondences.
    """

    def __init__(self, K):
        # Initiate ORB detector and specify the number of feaures to retain.
        self.orb = cv2.ORB_create(2500)

        # Feature matching using FLANN-based Matcher. Specify the default
        # parameters for the algorithm.
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1)
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(
            indexParams=index_params,
            searchParams=search_params)

        # Camera's intrinsic Matrix
        self.K = K

    def match_image_pairs(self, img1, img2):
        """
        This function detects and computes keypoints and descriptors from the query (I[t-1])
        and train (I[t]) image using ORB. Afterward, it establishes the correspondences between
        the images and returns the location of good keypoints in both images.
        Parameters
        ----------
        img1 (ndarray): query image
        img2 (ndarray): train image
        Returns
        -------
        q1 (ndarray): The good keypoints matches position in query image
        q2 (ndarray): The good keypoints matches position in train image
        """
        # Detect the keypoints and compute descriptors using ORB.
        query_keypoints, query_descriptors = self.orb.detectAndCompute(
            img1, None)
        train_keypoints, train_descriptors = self.orb.detectAndCompute(
            img2, None)

        # Find matches, two nearest neighbors.
        matches = self.flann.knnMatch(
            query_descriptors, train_descriptors, k=2)

        # Apply Lowe's ratio test to reject the false positives.
        ratio_threshold = 0.85
        good_matches = []

        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        # Visualize matches
        # img3 = cv2.drawMatches(
            # img1,
            # query_keypoints,
            # img2,
            # train_keypoints,
            # good_matches,
            # None,
            # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3)
        # plt.show()

        # Get the location of key points in both frames based on good_matches.
        q1 = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches])
        q2 = np.float32([train_keypoints[m.trainIdx].pt for m in good_matches])

        return q1, q2

    def estimate_pose(self, q1, q2):
        """This function estimates the relative pose between two images.
        It first calculates the essential matrix E using the image correspondences,
        then decomposes E to obtain the rotation matrix R and the translation vector t.
        Parameters
        -------
        q1 (ndarray): The good keypoints matches position in query image
        q2 (ndarray): The good keypoints matches position in train image
        Returns
        -------
        R : 3 x 3 rotation matrix
        t : 3 x 1 translation vector (direction)
        """
        # Calculate the essential matrix E from the corresponding points in two
        # images (q1,q2)
        E, _ = cv2.findEssentialMat(
            q1, q2, self.K, cv2.RANSAC, 0.999, 1.0, None)
        # Recovers the relative camera rotation and the translation from the estimated essential
        # matrix
        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K)

        return R, t


def main():
    # Load the camera calibration parameters
    parameters = []
    file_handle = open('camera_info/intrinsics.txt', encoding="utf-8")
    for line in file_handle:
        if line.startswith("#"):
            continue
        parameters.append(np.fromstring(line, dtype=np.float64, sep=' '))

    file_handle.close()
    # Projection Matrix
    P = np.array(parameters).reshape(3, 4)
    # Intrinsic Matrix
    K = P[0:3, 0:3]

    # create an object of image loader class
    location = 'image_sequence/'
    file_pattern = '000%d.jpg'
    start_id = 0  # first image number
    end_id = 200  # last  image number
    load_image = LoadImages(location, file_pattern, start_id, end_id)

    previous_image = None
    # create an object of VisualOdometry odometry class
    visual_odo = VisualOdometry(K)

    # Initialize the pose for the first frame
    current_position = np.zeros((3, 1))
    current_rotation = np.eye(3)
    trajectory = current_position.ravel().copy()

    # Load Ground Truth Data
    cols = ["t_x", "t_y", "t_z"]
    ground_truth_pos = pd.read_csv(
        'ground_truth_poses/trajectory.csv', usecols=cols)

    # Optional: I will provide the results for both cases.
    # Since GT is given, we can also use it to estimate the abs. scale for aligning
    # the predicted trajectory with the ground truth trajectory
    gt_xy_pos = ground_truth_pos[["t_x", "t_y"]].to_numpy()
    estimate_scale_info = np.linalg.norm(
        gt_xy_pos[:-1, :] - gt_xy_pos[1:, :], axis=1)
    # Set this flag to True to incorporate scale info. from GT
    scale_provided = False

    # Iterate over all images and estimate pose from consecutive images.
    for i in range(len(load_image)):
        ok, image = load_image.next()
        assert ok

        if previous_image is None:
            previous_image = image.copy()
            continue

        # Feature extraction and matching
        q1, q2 = visual_odo.match_image_pairs(previous_image, image)

        # Estimate R,t from image correspondences (q1,q2)
        R, t = visual_odo.estimate_pose(q1, q2)

        if scale_provided:
            scale = estimate_scale_info[i - 1]
        else:
            scale = 1.0

        # Update current position
        current_position += current_rotation.dot(t) * scale
        current_rotation = R.dot(current_rotation)

        # Append current_position to the trajectory for plotting purposes
        trajectory = np.vstack((trajectory, current_position.ravel()))
        # Set current image I[t] to previous image I[t-1]
        previous_image = image.copy()

    # Visualize Position Trajectory
    plt.plot(
        ground_truth_pos["t_x"],
        ground_truth_pos["t_y"],
        'r-',
        label='GT')
    plt.plot(-1 * trajectory[:, 2], -1 * trajectory[:, 0],
             'g-', label='Predicted (without scale info)')

    plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.xlabel('X [m]', fontsize=16)
    plt.ylabel('Y [m]', fontsize=16)
    plt.title('Monocular Visual Odometry', fontsize=16)
    # plt.savefig('trajectory_with_scale.jpg', dpi=900, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
