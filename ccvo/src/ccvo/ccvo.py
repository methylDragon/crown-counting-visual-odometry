import mos.ops
import mos.utils

import logging
from collections import deque

import numpy as np
import cv2

__all__ = [
    'CCVO',
    'default_config'
]

default_config = {
    'enable': {
        'visualisation': True,
        'preprocessing': True,

        'masking': True,
        'static_mask': True,

        'static_centroid_distance_estimation': False,
        'static_circle_diameter_estimation': False,

        'static_crossing_line': False
    },

    'mask_config': {
        'static_mask_config': {
            'mask_offsets': [110, 140, 0, 0] # Top, Bottom, Left, Right
        },

        'dynamic_mask_config': {
            'min_lines': 5,
            'canny_params': {'blur_amount': 15,
                             'min_thresh': 1400,
                             'max_thresh': 2511,
                             'aperture': 7,
                             'l2_grad': 1},
            'hough_line_params': {'rho': 1,
                                  'theta': 0.02,
                                  'threshold': 120,
                                  'min_line_length': 62,
                                  'max_line_gap': 50},
            'erode_params': {'kernel': np.ones((15,15), np.uint8),
                             'iterations': 5}
        }
    },

    'circle_detection_config': {
        'canny_params': {'blur_amount': 23,
                         'min_thresh': 3245,
                         'max_thresh': 2501,
                         'aperture':7,
                         'l2_grad': 1},
        'hough_circles_params': {'inverse_resolution': 1.8,
                                 'min_distance': 185,
                                 'canny_upper': 100,
                                 'threshold': 60,
                                 'min_radius': 50,
                                 'max_radius': 128}
    },

    'circle_tracking_config': {
        'velocity_thresh': 6, # For updating direction
        'confidence_thresh': 3, # For filtering centroids
        'size_estimator_window': 10, # To estimate size
        'static_diameter': 150,

        'centroid_tracking_config': {'max_missing_count': 3,
                                     'max_distance': 75,
                                     'smoothed_vel_window': 10}
    },

    'circle_counting_config': {
        'velocity_thresh': 6, # For filtering centroids
        'confidence_thresh': 3, # For filtering centroids
        'forward_dir': np.array([1, 0]),

        'count_mask_params': {'start_offset': 75,
                              'end_offset': 10000,
                              'thickness': 150}
    },

    'centroid_distance_estimation_config': {
        'static_distance': 320,
        'distance_estimator_window': 10
    }
}

# CCVO =========================================================================
class CCVO:
    """Algorithm for Crown Counting Visual Odometry (CCVO)."""
    def __init__(self, config={}):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.centroid_tracker = None

        self.config = default_config
        self.configure(config)

        self.centroid_tracker = mos.utils.CentroidTracker(
            **self.config['circle_tracking_config']['centroid_tracking_config']
        )

        # Centroid Statistics
        self.centroid_statistics = None
        self.average_vel = None
        self.thresholded_average_vel = None

        # Centroid pixel distance estimators
        estimator_config = self.config['centroid_distance_estimation_config']
        self.counted_centroid_distances = deque(
            [estimator_config['static_distance']],
            maxlen=estimator_config['distance_estimator_window']
        )
        self.counted_centroid_distance = estimator_config['static_distance']
        self.estimated_centroid_distance = estimator_config['static_distance']

        # Circle pixel radius estimators
        tracking_config = self.config['circle_tracking_config']
        self.circle_diameters = deque(
            [tracking_config['static_diameter']],
            maxlen=tracking_config['size_estimator_window']
        )
        self.estimated_circle_diameter = tracking_config['static_diameter']

        # Count Statistics
        self.counted_ids = set()
        self.last_counted_id = -1
        self.last_counted_vect = None
        self.reversed = False

        self.count = 0
        self.subcount = 0

        # Image Statistics
        self.img_height, self.img_width = (0, 0)

        # Visualisation
        self.visualisation_images = {}

    def get_info(self):
        return {
            'count': self.count,
            'subcount': self.subcount,
            'reversed': self.reversed,
            'average_pixel_vel': self.average_vel,
            'estimated_circle_diameter': self.estimated_circle_diameter,
            'estimated_centroid_distance': self.estimated_centroid_distance
        }

    def get_visualisations(self):
        return self.visualisation_images

    def update(self, image):
        self.img_height, self.img_width = image.shape[:2]

        if self.config['enable']['preprocessing']:
            image = mos.ops.preprocess_clahe_blur(image)

        if self.config['enable']['masking']:
            mask = self._mask(image)
            if mask is not None:
                masked_img = cv2.bitwise_and(mask, image)
        else:
            masked_img = image
            mask = None

        circles = self._detect_and_track_circles(mask, masked_img)
        self._count_circles(circles, image)

    # CONFIGURE ================================================================
    def configure(self, update):
        configured = tree_update(self.config, update, self.logger)
        self._configure_centroid_tracker()
        return configured

    def _configure_centroid_tracker(self):
        if self.centroid_tracker is not None:
            config = self.config['circle_tracking_config']
            config = config['centroid_tracking_config']

            self.centroid_tracker.max_missing_count = config['max_missing_count']
            self.centroid_tracker.max_distance = config['max_distance']

            # NOTE: Velocity window cannot be updated

    # MASK =====================================================================
    def _mask(self, image):
        """Generate mask for image either statically or dynamically."""
        if self.config['enable']['static_mask']: # With a static mask
            masked_img = self._static_mask(image)
        else: # By finding union of widest parallel lines
            masked_img = self._dynamic_mask(image)

        # Save visualisation images
        if self.config['enable']['visualisation']:
            self.visualisation_images['masked_img'] = masked_img

        return masked_img

    def _static_mask(self, image):
        config = self.config['mask_config']['static_mask_config']
        mask = np.zeros(image.shape, np.uint8)

        top, bottom, left, right = config['mask_offsets']

        return cv2.fillConvexPoly(
            mask,
            np.array([
                [left, top],
                [self.img_width - right, top],
                [self.img_width - right, self.img_height - bottom],
                [left, self.img_height - bottom]
            ]),
            (255, 255, 255)
        )

    def _dynamic_mask(self, image, config):
        config = self.config['mask_config']['dynamic_mask_config']

        canny = mos.ops.canny(image, **config['canny_params'])
        lines = mos.ops.hough_lines_probabilistic(
            canny, **config['hough_line_params']
        )

        if self.config['enable']['visualisation']:
            self.visualisation_images['dynamic_mask_lines'] = \
                mos.ops.draw_hough_lines(np.copy(image), lines)

        # Skip if too little lines
        if (lines is not None and len(lines) > config['min_lines']):
            hull_img = np.zeros(image.shape, np.uint8)

            # Mask image by finding union of widest parallel lines
            mask = mos.ops.get_mask_from_parallel_lines(hull_img, lines)
            return cv2.erode(mask, **config['erode_params'])
        else:
            return image

    # DETECT AND TRACK CIRCLES =================================================
    def _detect_and_track_circles(self, mask, image):
        # Detect Circles =======================================================
        config = self.config['circle_detection_config']

        # Detect canny edges and mask
        circle_canny = mos.ops.canny(image, **config['canny_params'])

        if mask is not None:
            masked_canny = cv2.bitwise_and(
                cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), circle_canny
            )
        else:
            masked_canny = circle_canny

        # Detect circles
        circles = mos.ops.hough_circles(
            masked_canny, **config['hough_circles_params']
        )
        boxes = mos.utils.circles_to_boxes(circles)

        # Track Circle Centroids ===============================================
        config = self.config['circle_tracking_config']

        try:
            self.centroid_statistics = self.centroid_tracker.update(
                boxes[:].reshape(-1, 4)
            )

            confidences, smoothed_vels, rects = (
                self.centroid_statistics['confidences'].values(),
                self.centroid_statistics['smoothed_velocities'].values(),
                self.centroid_statistics['rects'].values()
            )

            # Update circle size estimates
            filtered_rects = [
                rect for
                rect, conf in zip(rects, confidences)
                if conf > config['confidence_thresh']
            ]

            self._estimate_circle_diameter(filtered_rects)

            # Compute velocities
            filtered_vels = [
                smoothed_vel for
                smoothed_vel, conf in zip(smoothed_vels, confidences)
                if conf > config['confidence_thresh']
            ]

            if len(filtered_vels) > 0:
                average_vel = np.mean(filtered_vels, axis=0)
            else:
                average_vel = np.array([0, 0])

            # Handle reversals and velocity rectification ======================
            if self.config['enable']['static_crossing_line']:
                forward = self.config['circle_counting_config']['forward_dir']
                forward_dir = forward / np.linalg.norm(forward)

                # Check for reversals
                if np.linalg.norm(average_vel) > config['velocity_thresh']:
                    if np.dot(self.thresholded_average_vel, forward_dir) < 0:
                        if not self.reversed:
                            self.reversed = True
                            self.counted_ids = set()
                    elif self.reversed:
                        self.reversed = False
                        self.counted_ids = set()

                self.thresholded_average_vel = average_vel

                # Rectify average velocity
                self.average_vel = np.multiply(average_vel, forward_dir)
            else:
                # Check for reversals
                if np.linalg.norm(average_vel) > config['velocity_thresh']:
                    if self.thresholded_average_vel is not None:
                        if np.dot(self.thresholded_average_vel, average_vel) < 0:
                            self.reversed ^= 1
                            self.counted_ids = set()

                    self.thresholded_average_vel = average_vel

                # Rectify average velocity
                if np.linalg.norm(average_vel) > config['velocity_thresh'] / 2:
                    if self.reversed:
                        self.average_vel = -abs(average_vel)
                    else:
                        self.average_vel = abs(average_vel)
                else:
                    self.average_vel = average_vel

        except Exception as e:
            self.logger.error(e)

        # Save Visualisation Images ============================================
        if self.config['enable']['visualisation']:
            self.visualisation_images['circle_detection_canny'] = \
                cv2.cvtColor(masked_canny, cv2.COLOR_GRAY2BGR)

            self.visualisation_images['ccvo'] = \
                mos.ops.draw_hough_circles(
                    cv2.cvtColor(masked_canny, cv2.COLOR_GRAY2BGR), circles
                )

        return circles

    # COUNT CIRCLE CENTROIDS ===================================================
    def _count_circles(self, circles, image):
        config = self.config['circle_counting_config']

        centroids, confidences, smoothed_vels, vels = (
            self.centroid_statistics['centroids'],
            self.centroid_statistics['confidences'],
            self.centroid_statistics['smoothed_velocities'],
            self.centroid_statistics['velocities']
        )

        if self.thresholded_average_vel is not None:
            # Compute directions ===============================================
            if self.config['enable']['static_crossing_line']:
                forward_dir = self.config['circle_counting_config']['forward_dir']

                if self.reversed:
                    direction = -forward_dir / np.linalg.norm(forward_dir)
                else:
                    direction = forward_dir / np.linalg.norm(forward_dir)
            else:
                direction = (self.thresholded_average_vel
                             / np.linalg.norm(self.thresholded_average_vel))

            # Compute end points for count mask
            dir_x, dir_y = (np.array([self.img_width/2, self.img_height/2])
                            - config['count_mask_params']['start_offset']
                            * direction)

            end_x, end_y = (np.array([self.img_width/2, self.img_height/2])
                            - config['count_mask_params']['end_offset']
                            * direction)

            # Draw count mask (not for visualisation!) =========================
            count_mask = np.zeros(image.shape[:2], np.uint8)
            cv2.line(count_mask,
                     (int(dir_x), int(dir_y)),
                     (int(end_x), int(end_y)),
                     color=(255, 255, 255),
                     thickness=config['count_mask_params']['thickness'],
                     lineType=cv2.LINE_AA)

            # Draw crossing line using average velocity
            if self.config['enable']['visualisation']:
                self.visualisation_images['count_mask'] = count_mask

                self._draw_crossing_line(
                    direction, dir_x, dir_y
                )

        # Count centroids ======================================================
        for centroid_id in centroids.keys():
            # Filter out low confidence centroids
            if confidences[centroid_id] < config['confidence_thresh']:
                continue

            centroid = centroids[centroid_id]

            # Draw centroids
            if self.config['enable']['visualisation']:
                cv2.putText(
                    self.visualisation_images['ccvo'],
                    f"ID {centroid_id} (conf:{confidences[centroid_id]})",
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                cv2.circle(self.visualisation_images['ccvo'],
                           (centroid[0], centroid[1]),
                           4, (0, 255, 0), -1)

            if centroid_id in self.counted_ids: # Skip counted centroids
                continue

            count_x, count_y = centroids[centroid_id]

            try:
                # Count a centroid if it is:
                # - Moving in the same direction as average,
                # - Is in the count mask
                # - Is moving fast enough
                if (np.dot(smoothed_vels[centroid_id],
                           self.thresholded_average_vel) > 0
                        and count_mask[count_y, count_x] > 0
                        and np.linalg.norm(vels[centroid_id])
                            > config['velocity_thresh']):

                    self.counted_ids.add(centroid_id)

                    if self.reversed:
                        self.count -= 1
                    else:
                        self.count += 1

                    # Append
                    self.last_counted_id = centroid_id
                    self._estimate_centroid_distance(direction)

            except Exception as e:
                pass

        # Compute Subcount ====================================================
        if self.thresholded_average_vel is not None:
            self._compute_last_count_distances(direction)
            self._compute_subcount()

        # Save Visualisation Images ============================================
        if self.config['enable']['visualisation']:
            cv2.putText(self.visualisation_images['ccvo'],
                        f"Count: {self.count}",
                        (10, self.img_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(self.visualisation_images['ccvo'],
                        f"Subcount: {self.subcount}",
                        (10, self.img_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(self.visualisation_images['ccvo'],
                        f"Average Pixel Vel: {self.average_vel}",
                        (10, self.img_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw line to latest counted centroid
            if self.last_counted_id in centroids:
                # Draw direction arrow
                cv2.line(self.visualisation_images['ccvo'],
                         (self.img_width // 2, self.img_height // 2),
                         (centroids[self.last_counted_id][0],
                          centroids[self.last_counted_id][1]),
                         color=(0, 255 * (self.reversed ^ 1), 255 * self.reversed),
                         thickness=2)

    # ESTIMATORS AND COMPUTATIONS ==============================================
    def _estimate_circle_diameter(self, rects):
        config = self.config['circle_detection_config']

        for rect in rects:
            self.circle_diameters.append(abs(rect[0] - rect[1]))

        if self.config['enable']['static_circle_diameter_estimation']:
            self.estimated_circle_diameter = config['static_diameter']
        else:
            self.estimated_circle_diameter = np.mean(self.circle_diameters)

    def _estimate_centroid_distance(self, direction):
        config = self.config['centroid_distance_estimation_config']

        if self.thresholded_average_vel is not None:
            self._compute_last_count_distances(direction)

        self.counted_centroid_distances.append(self.counted_centroid_distance)
        self.counted_centroid_distance = 0

        if self.config['enable']['static_centroid_distance_estimation']:
            self.estimated_centroid_distance = config['static_distance']
        else:
            self.estimated_centroid_distance = np.mean(
                self.counted_centroid_distances
            )


    def _compute_last_count_distances(self, direction):
        centroids = self.centroid_statistics['centroids']

        if self.last_counted_id not in centroids:
            return

        # Compute component of vector from image centre to last-counted centroid
        # that is along the average direction of travel
        self.last_counted_vect = np.dot(
            ((self.img_width // 2, self.img_height // 2)
             - centroids[self.last_counted_id]),
            direction
        )

        # Keep track of the maximum magnitude of this vector
        self.counted_centroid_distance = max(
            self.counted_centroid_distance,
            np.linalg.norm(self.last_counted_vect)
        )

    def _compute_subcount(self):
        centroids = self.centroid_statistics['centroids']

        # Only update subcount if we can actually see the last centroid
        if self.last_counted_id not in centroids:
            return

        subcount = mos.utils.clamp(
            self.last_counted_vect / self.estimated_centroid_distance,
            smallest = -1,
            largest = 1
        )

        if self.reversed:
            self.subcount = -subcount
        else:
            self.subcount = subcount

    # DRAWING ==================================================================
    def _draw_crossing_line(self, direction, dir_x, dir_y):
        normal_vel = np.array([direction[1], -direction[0]])
        crossing_line = mos.utils.LinearLine.from_vector(
            (self.img_width/2, self.img_height/2), normal_vel
        )

        crossing_line_points = crossing_line.intercepts_with_bounding_box(
            ([0, 0], [self.img_width, self.img_height])
        ).astype(np.int32)

        # Draw crossing line
        cv2.line(self.visualisation_images['ccvo'],
                 tuple(crossing_line_points[0]),
                 tuple(crossing_line_points[1]),
                 color=(0, 0, 255),
                 thickness=2)

        # Draw direction arrow
        cv2.arrowedLine(self.visualisation_images['ccvo'],
                        (self.img_width // 2, self.img_height // 2),
                        (int(dir_x), int(dir_y)),
                        color=(0, 255 * (self.reversed ^ 1), 255 * self.reversed),
                        thickness=4)

# MISC UTILITIES ===============================================================
# Tree update function for updating config
def tree_update(tree, update, logger=None):
    """
    Update a nested dictionary while preserving structure and type.

    In the event of a key clash:
        - Top level keys will mask lower level keys.
        - Lower level keys will mask either non-deterministically or in order.

    If you need to update lower level keys, update the top level key dict
    instead.
    """
    configured = False

    for key, value in update.items():
        # If update key is in the top level
        if key in tree:
            if type(value) is not type(tree[key]):
                if logger:
                    logger.warning(f"INVALID CONFIGURATION TYPE FOR: "
                                   f"({repr(key)}, {repr(value)}). "
                                   f"Expected {type(tree[key]).__name__}.")
                continue

            if type(tree[key]) is dict:
                for k, v in value.items():
                    if tree_update(tree[key], {k: v}, logger):
                        configured = True
            else:
                tree.update({key: value})
                configured = True

            continue

        # If update key is in a lower level
        branch_configured = False
        for branch in tree.values():
            if type(branch) is dict:
                if tree_update(branch, {key: value}, logger):
                    branch_configured = True
                    configured = True
                    break

        if branch_configured:
            continue

        if logger:
            logger.warning(
                f"INVALID UPDATE KEY: {repr(key)}, KEY NOT IN CONFIG"
            )

    return configured
