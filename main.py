import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN
import math


def extract_frames(video_path, output_folder):
    cam = cv2.VideoCapture(video_path)
    frameno = 0

    while True:
        ret, frame = cam.read()
        if ret:
            name = os.path.join(output_folder, str(frameno) + ".jpg")
            print("New frame captured:", name)
            cv2.imwrite(name, frame)
            frameno += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


def edge_detector(image):
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )  # converts the image into black and white
    filtered = cv2.medianBlur(
        gray, ksize=5
    )  # used to reduce the noise : replaces each pixel's value with the median value of its neighboring pixel
    scharr_x = cv2.Scharr(
        filtered, cv2.CV_64F, 1, 0
    )  # cal the gradient of the intensity in x dir
    scharr_y = cv2.Scharr(
        filtered, cv2.CV_64F, 0, 1
    )  # cal the gradient of the intensity in y dir
    edges = cv2.magnitude(
        scharr_x, scharr_y
    )  # computes the magnitude of the gradient at each pixel
    threshold_value = 150
    _, binary_edges = cv2.threshold(
        edges, threshold_value, 255, cv2.THRESH_BINARY
    )  # applies a binary threshold to the magnitude to remove anything below it
    return binary_edges


def erosion_dilation(binary_edges):
    kernel_size = 3  # kernal size here is 3*3
    kernel = np.ones(
        (kernel_size, kernel_size), np.uint8
    )  # creates a kernal size 3*3 filled with ones
    eroded_edges = cv2.erode(
        binary_edges, kernel, iterations=1
    )  # erosion to remove small moise (sets the kernel center to min pixel value)
    dilated_edges = cv2.dilate(
        eroded_edges, kernel, iterations=1
    )  # dilation to enhance the edges (sets the kernel center to man pixel value)
    return dilated_edges


def connecting_components(binary_edges, offset):
    height, width = binary_edges.shape  # dim of the image
    roi_right = binary_edges[
        :, width // 2 + offset :
    ]  # want to perform it on the right half on the image only (offset = right to the wing )

    labeled_regions = label(
        roi_right
    )  # label function applies a unique label to each component
    properties = regionprops(
        labeled_regions
    )  #  regionprops function cal properties of the labeled regions such as centroid, area, bbox
    centroids = []

    for (
        prop
    ) in (
        properties
    ):  # loop iterates over the list of properties and for each connected components, extracts the centroid's row and col cor-ordinates using prop.centroid
        centroid_row, centroid_col = prop.centroid
        centroids.append([centroid_row, centroid_col])
        # area = prop.area
        # Draw centroid on the right side by adding width // 2 + offset to the x-coordinate
        # cv2.circle(image, (int(centroid_col) + width // 2 + offset, int(centroid_row)), 5, (0, 0, 255), -1)

    return np.array(centroids)  # returning the centroid values


def dbscan(centroids):
    if len(centroids) > 0:
        eps = 10  # max distance between 2 samples to be considered neighbours
        min_samples = 1
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(
            centroids
        )  # fiting DBSCAN model on centroids data
        cluster_labels = (
            dbscan.labels_
        )  # that it belongs to which cluster (maybe I dont need this)
        mean_centroid = np.mean(
            centroids, axis=0
        )  # cal the mean of the centroids row wise(do i need to do dbscan? i can get this without it)
        average_distance = np.mean(
            np.linalg.norm(centroids - mean_centroid, axis=1)
        )  # cal the eucledian distance between the meaan centroid and all the other centroids
        return cluster_labels, mean_centroid, average_distance
    return None, None, None


def draw_result(image, cluster_labels, mean_centroid, average_distance):
    if mean_centroid is not None:
        height, width, _ = image.shape
        offset = 160  # offset to perfprm operation right on the wing
        # Draw pink circle approximating the cluster area
        cluster_center = (
            int(mean_centroid[1]) + width // 2 + offset,
            int(mean_centroid[0]),
        )  # dim of the center cluster
        cv2.circle(
            image, cluster_center, int(average_distance), (255, 192, 203), 2
        )  # circle around the clusters
        # Display radius values
        radius_text = f"Radius: {int(average_distance)}"
        cv2.putText(
            image,
            radius_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 192, 203),
            2,
        )
        # Draw red dot at the mean centroid
        cv2.circle(image, cluster_center, 5, (0, 0, 255), -1)  # vortex center
        # Calculate and draw the horizontal component line
        horizontal_distance = abs(int(cluster_center[0]) - (width // 2 + offset))
        cv2.line(
            image,
            (width // 2 + offset, int(mean_centroid[0])),
            cluster_center,
            (0, 255, 0),
            2,
        )
        # Display horizontal distance value on the horizontal component line
        cv2.putText(
            image,
            f"Horizontal Distance: {horizontal_distance}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return image, cluster_center


def dist(cluster_centers):
    distances = [0]

    for i in range(len(cluster_centers) - 1):
        x1, y1 = cluster_centers[i]
        x2, y2 = cluster_centers[i + 1]

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(dist)

    print(distances)
    len(distances)
    return distances


def dist_horizontal(cluster_centers):
    distances = [0]

    for i in range(len(cluster_centers) - 1):
        x1, _ = cluster_centers[i]
        x2, _ = cluster_centers[i + 1]

        dist = abs(x2 - x1)
        distances.append(dist)

    # print(distances)
    return distances


def assign_ids(image, cluster_centers, frame_number, frame_rate):
    ids = [0]  # Start with the first ID
    distances = dist_horizontal(cluster_centers)
    cumulative_distances = {0: 0.0}  # Initialize cumulative distances with ID 0
    frame_duration_seconds = 0.0  # frame_number / frame_rate

    for i, distance in enumerate(distances):
        if cluster_centers[i][0] < 1500 and distance > 65:
            ids.append(ids[-1] + 1)  # Assign a new ID
            cumulative_distances[
                ids[-1]
            ] = 0.0  # Reset the cumulative distance for the new ID
            frame_duration_seconds = 0.0
        else:
            ids.append(ids[-1])  # Retain the same ID
            cumulative_distances[
                ids[-1]
            ] += distance  # Accumulate the distance for the current ID

        frame_duration_seconds = round(
            frame_duration_seconds + 0.1, 2
        )  # Increment frame duration by 0.1 seconds

    # Keeping only the last ID and removing everything else
    if len(ids) > 1:
        ids = ids[-1:]
        cluster_centers = cluster_centers[-1:]

    # Printing the IDs - cumulative distances and duration

    for i, (cluster_center, id) in enumerate(zip(cluster_centers, ids)):
        cv2.putText(
            image,
            f"ID: {id} - Distance: {cumulative_distances[id]:.2f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Duration: {frame_duration_seconds:.2f}s",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    print(list(zip(cluster_centers, cumulative_distances, ids)))
    return image, list(zip(cluster_centers, ids))


def process_frame(frame_number, cluster_centers):
    if frame_number > 40:
        return None, cluster_centers  # Skip frames after the first 40

    image_path = "/content/frames/" + str(frame_number) + ".jpg"
    image = cv2.imread(image_path)

    if image is not None:
        binary_edges = edge_detector(image)
        dilated_edges = erosion_dilation(binary_edges)
        centroids = connecting_components(dilated_edges, 160)
        cluster_labels, mean_centroid, average_distance = dbscan(centroids)
        if mean_centroid is not None:
            image, cluster_center = draw_result(
                image, cluster_labels, mean_centroid, average_distance
            )
            cluster_centers.append(cluster_center)

            # Draw IDs in the current frame
            image, ids = assign_ids(image, cluster_centers)
            return image, cluster_centers
        else:
            print(f"No centroids found for frame: {frame_number}")
            return None, cluster_centers
    else:
        print(f"Image not found: {image_path}")
        return None, cluster_centers


def create_video():
    num_frames = 40  # Test on the first 40 frames
    frame_folder = "/content/frames"
    output_folder = "/content/output"
    os.makedirs(
        output_folder, exist_ok=True
    )  # creates output folder if it dosen't exist

    output_filename = "50%_Full_Clap_Vortex_Tracking_Test_3.mp4"
    output_path = os.path.join(
        output_folder, output_filename
    )  # complete path for the output

    if os.path.exists(output_path):  # if file with the same name exist, it is removed
        os.remove(output_path)

    frame_width = 1920
    frame_height = 1080
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame_width, frame_height)
    )  # creates the output video (10 fps, should change it to 24 fps)

    cluster_centers = []

    for frame_number in range(
        1, num_frames + 1
    ):  # iterates through the frame (it should be 0, num_frames +1 ?)
        processed_frame, cluster_centers = process_frame(
            frame_number, cluster_centers
        )  # process_frames_test is called for each frame
        if processed_frame is not None:
            out.write(processed_frame)  # written to the output video
        else:
            print(f"Frame not found: {frame_number}.jpg")

    out.release()


if __name__ == "__main__":
    # Provide the path to the video and the output folder
    video_path = os.path.join("content", "50%_Full_Clap.mp4")
    output_folder = os.path.join("content", "frames")

    os.makedirs(output_folder, exist_ok=True)
    extract_frames(video_path, output_folder)

    create_video()
