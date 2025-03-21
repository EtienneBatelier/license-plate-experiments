import numpy as np
from PIL import Image
from FH_segmentation import graph_segmentation, merge_small_components
import visualizing_tools as vt


### Utility

def partition_to_bboxes(pixel_partition, height, width):
    bboxes_dict = {}
    for y in range(height):
        for x in range(width):
            rep = pixel_partition.find(y*width + x)
            if rep not in bboxes_dict:
                bboxes_dict[rep] = [[x, y, x, y]]
            else:
                bbox = bboxes_dict[rep]
                if x < bbox[0][0]:
                    bboxes_dict[rep][0][0] = x
                if y < bbox[0][1]:
                    bboxes_dict[rep][0][1] = y
                if x > bbox[0][2]:
                    bboxes_dict[rep][0][2] = x
                if y > bbox[0][3]:
                    bboxes_dict[rep][0][3] = y
    return bboxes_dict

def bbox_dict_to_list(bboxes_dict):
    bboxes = []
    for key in bboxes_dict:
        for bbox in bboxes_dict[key]:
            bboxes.append(bbox)
    return bboxes


### Similarity measures

def dissimilarity(red_1, red_2, green_1, green_2, blue_1, blue_2):
    return int(np.abs(red_1 - red_2) + np.abs(green_1 - green_2) + np.abs(blue_1 - blue_2))
    #return (red_channel[y1, x1] - red_channel[y2, x2])**2 + \
    #    (green_channel[y1, x1] - green_channel[y2, x2])**2 + \
    #    (blue_channel[y1, x1] - blue_channel[y2, x2])**2

def RGB_histograms(RGB_image, pixel_partition, height, width, number_bins = 25):
    red_histograms, green_histograms, blue_histograms = {}, {}, {}
    color_histograms = [red_histograms, green_histograms, blue_histograms]
    for i in range(3):
        channel = RGB_image[:, :, i]
        histograms = color_histograms[i]
        for y in range(height):
            for x in range(width):
                rep = pixel_partition.find(y * width + x)
                if rep in histograms:
                    histograms[rep].append(channel[y, x])
                else:
                    histograms[rep] = [channel[y, x]]
    for i in range(3):
        histograms = color_histograms[i]
        for rep in histograms:
            histograms[rep] = np.histogram(histograms[rep],
                                           bins=number_bins, range=(0, 255), density=True)[0]
    return color_histograms

def histogram_inter(color_histograms_1, color_histograms_2):
    sum_ = 0.
    for i in range(3):
        for j in range(len(color_histograms_1[i])):
            sum_ += min(color_histograms_1[i][j], color_histograms_2[i][j])
    return sum_


### Selective search algorithm

def initial_segmentation(RGB_image, k):
    # Turn the image into a weighted graph
    height, width = RGB_image.shape[:2]
    red_channel = RGB_image[:, :, 0]
    green_channel = RGB_image[:, :, 1]
    blue_channel = RGB_image[:, :, 2]
    edges = np.zeros(shape = (width*height*4, 3), dtype = int)
    i = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges[i, 0] = int(y*width + x)
                edges[i, 1] = int(y*width + (x + 1))
                edges[i, 2] = dissimilarity(red_channel[y, x], red_channel[y, x + 1],
                                            green_channel[y, x], green_channel[y, x + 1],
                                            blue_channel[y, x], blue_channel[y, x + 1])
                i += 1
            if y < height - 1:
                edges[i, 0] = int(y*width + x)
                edges[i, 1] = int((y + 1)*width + x)
                edges[i, 2] = dissimilarity(red_channel[y, x], red_channel[y + 1, x],
                                            green_channel[y, x], green_channel[y + 1, x],
                                            blue_channel[y, x], blue_channel[y + 1, x])
                i += 1
            if (x < width - 1) and (y < height - 2):
                edges[i, 0] = int(y*width + x)
                edges[i, 1] = int((y + 1)*width + (x + 1))
                edges[i, 2] = dissimilarity(red_channel[y, x], red_channel[y + 1, x + 1],
                                            green_channel[y, x], green_channel[y + 1, x + 1],
                                            blue_channel[y, x], blue_channel[y + 1, x + 1])
                i += 1
            if (x < width - 1) and (y > 0):
                edges[i, 0] = int(y*width + x)
                edges[i, 1] = int((y - 1)*width + (x + 1))
                edges[i, 2] = dissimilarity(red_channel[y, x], red_channel[y - 1, x + 1],
                                            green_channel[y, x], green_channel[y - 1, x + 1],
                                            blue_channel[y, x], blue_channel[y - 1, x + 1])
                i += 1

    # Return segmented graph
    return graph_segmentation(width*height, edges, k)

def hierarchical_grouping(RGB_image, pixel_partition, edges):
    # Record histograms for each region and each channel
    height, width = RGB_image.shape[:2]
    color_histograms = RGB_histograms(RGB_image, pixel_partition, height, width, 15)

    # Update the edge weights so that they record region similarity (= histogram intersection)
    edges_new_weights = []
    for edge in edges:
        a, b = pixel_partition.find(edge[0]), pixel_partition.find(edge[1])
        if a != b:
            similarity = histogram_inter([color_histograms[i][a] for i in range(3)],
                                         [color_histograms[i][b] for i in range(3)])
            edges_new_weights.append([a, b, similarity])

    # Extracting initial bounding boxes
    bboxes_dict = partition_to_bboxes(pixel_partition, height, width)

    # Grouping components
    max_sim_idx = np.argmax([edge[2] for edge in edges_new_weights])
    while len(edges_new_weights) > 0:
        # Merge most similar pair of regions
        edge = edges_new_weights.pop(max_sim_idx)
        a, b = pixel_partition.find(edge[0]), pixel_partition.find(edge[1])
        if a != b:
            old_bbox_1, old_bbox_2 = bboxes_dict[a][-1], bboxes_dict[b][-1]
            new_bbox = [min(old_bbox_1[0], old_bbox_2[0]),
                        min(old_bbox_1[1], old_bbox_2[1]),
                        max(old_bbox_1[2], old_bbox_2[2]),
                        max(old_bbox_1[3], old_bbox_2[3])]
            pixel_partition.merge(a, b)
            new_a = pixel_partition.find(a)
            bboxes_dict[new_a].append(new_bbox)

            # Propagate histograms
            for i in range(3):
                histograms = color_histograms[i]
                size_a, size_b = pixel_partition.subset_size(a), pixel_partition.subset_size(b)
                for j in range(len(histograms[new_a])):
                    histograms[new_a][j] = (size_a*histograms[a][j] + size_b*histograms[b][j])/(size_a + size_b)

            # Update similarities and max_sim_idx
            i, max_sim_idx, max_sim = 0, 0, edges_new_weights[0][2]
            while i < len(edges_new_weights):
                a_, b_ = pixel_partition.find(edges_new_weights[i][0]), pixel_partition.find(edges_new_weights[i][1])
                if a_ == b_:
                    edges_new_weights.pop(i)
                    continue
                if a_ == new_a or b_ == new_a:
                    edges_new_weights[i] = [a_, b_, histogram_inter([color_histograms[i][a_] for i in range(3)],
                                                                    [color_histograms[i][b_] for i in range(3)])]
                if edges_new_weights[i][2] > max_sim:
                    max_sim_idx = i
                    max_sim = edges_new_weights[i][2]
                i += 1

    return bboxes_dict

def selective_search(PIL_image, k = 150, sigma = 0.8, min_size = 50):
    # Apply Gaussian filter to the image
    RGB_image = np.array(PIL_image)
    RGB_image = vt.RGB_Gaussian_filter(RGB_image, sigma)  # Warning: entries are floats (between 0 and 255)

    # Begin with the initial segmentation
    pixel_partition, edges = initial_segmentation(RGB_image, k)

    # Merge the very small components
    pixel_partition, edges = merge_small_components(pixel_partition, edges, min_size)

    # Group initial components into larger ones
    bboxes_dict = hierarchical_grouping(RGB_image, pixel_partition, edges)
    return bbox_dict_to_list(bboxes_dict)


### Test

def test_selective_search():
    # Open an image with PIL and resize it so that it has ~150k pixels
    #PIL_image = Image.open("./image_1.png")
    #PIL_image = Image.open("./image_2.jpg")
    PIL_image = Image.open("./image_3.png")
    PIL_image = vt.resize(PIL_image)
    
    # Algorithm parameters
    k = 200         # The higher k, the larger the initial regions
    sigma = 0.8     # Standard variation in Gaussian filter
    min_size = 100  # Minimum size of a region
    
    # Number of bounding boxes
    bboxes = selective_search(PIL_image, k, sigma, min_size)
    print("Number of bounding boxes: " + str(len(bboxes)))
    
    # Visualize the bounding boxes
    print("Creating visual")
    vt.plot_two_images(PIL_image, vt.draw_bboxes(PIL_image, bboxes))


#test_selective_search()