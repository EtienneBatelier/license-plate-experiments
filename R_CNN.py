import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
import os

import selective_search as ss
import visualizing_tools as vt
import NMS
import CNN


### R_CNN algorithm

def aspect_ratio_filter(bboxes):
    i = 0
    while i < len(bboxes):
        bbox = bboxes[i]
        if bbox[2] == bbox[0]:
            bboxes.pop(i)
        else:
            aspect_ratio_inverse = (bbox[3] - bbox[1])/(bbox[2] - bbox[0])
            if aspect_ratio_inverse < 1.2 or aspect_ratio_inverse > 2.2:
                bboxes.pop(i)
            else:
                i += 1
    return bboxes

def size_filter(bboxes):
    i = 0
    while i < len(bboxes):
        bbox = bboxes[i]
        area = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])
        if area < 100 or area > 18000:
            bboxes.pop(i)
        else:
            i += 1
    return bboxes

def ROI_estimations(PIL_image, bboxes, CNN_input_size = (30, 40),
                    CNN_file_path = "./pytorch_files/saved_models/lp_characters_trained.pt"):
    # Load the trained CNN
    trained_CNN = CNN.LPCharactersModel(35)
    trained_CNN.load_state_dict(torch.load(CNN_file_path, weights_only=True))
    trained_CNN.eval()

    # Convert the image to grayscale
    PIL_image_GS = PIL_image.convert("L")

    # Create regions of interest and pass them to the CNN
    width, height = PIL_image.size
    estimations = []
    for bbox in bboxes:
        # It is recommended to slightly enlarge the bbox to add context for CNN evaluation
        bbox_enlarged = [max(0, bbox[0] - 4),
                         max(0, bbox[1] - 4),
                         min(width, bbox[2] + 4),
                         min(height, bbox[3] + 4)]
        ROI = PIL_image_GS.crop(bbox_enlarged)
        ROI = ROI.resize(CNN_input_size)
        ROI_tensor = transforms.PILToTensor()(ROI).unsqueeze(0).float()
        estimation = torch.Tensor.detach(trained_CNN(ROI_tensor).squeeze())
        estimations.append(estimation)

    # The network gives logits. Convert to probabilities
    estimations = np.exp(np.array(estimations))
    estimations /= np.add(1, estimations)
    return estimations

def isolate_matches(bboxes, estimations, prob_threshold = None):
    matches = []
    if len(bboxes) == 0: return matches
    if prob_threshold is None:
        prob_threshold = np.quantile([x for xs in estimations for x in xs], 0.97)
    for j, bbox in enumerate(bboxes):
        #i = np.argmax(estimations[j])
        for i in range(35):
            if estimations[j][i] > prob_threshold:
                matches.append([j, i, estimations[j][i]])
    return matches

def R_CNN(PIL_image, k, sigma, min_size, iou_threshold, verbose = True, visualize = False):
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                  'V', 'W', 'X', 'Y', 'Z']
    PIL_image = vt.resize(PIL_image.convert("RGB"))

    if verbose: print("Generating regions of interest")
    bboxes = ss.selective_search(PIL_image, k, sigma, min_size)
    if verbose: print(str(len(bboxes)) + " bounding boxes generated")
    bboxes = size_filter(aspect_ratio_filter(bboxes))
    if verbose: print("Filtered down to " + str(len(bboxes)))

    # Make the network evaluate the ROI
    if verbose: print("Finding matches")
    estimations = ROI_estimations(PIL_image, bboxes)
    matches = isolate_matches(bboxes, estimations)
    if verbose: print(str(len(matches)) + " found")

    # Applying non-maximum suppression
    if verbose: print("Removing redundant matches with non-maximum suppression")
    matches, bboxes = NMS.NMS(matches, bboxes, iou_threshold)

    # Read the remaining matches
    left_to_right_idx = np.argsort([bbox[0] for bbox in bboxes])
    bboxes = [bboxes[i] for i in left_to_right_idx]
    matches = [matches[i] for i in left_to_right_idx]
    guess = ""
    for match in matches:
        guess += characters[match[1]]

    # Visualize the result
    if visualize:
        if verbose: print("Creating visual")
        vt.plot_two_images(PIL_image, vt.draw_bboxes(PIL_image, bboxes))

    return guess


### Test

def Levenshtein_dist(str_1, str_2):
    n, m = len(str_1), len(str_2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str_1[i - 1] == str_2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]

def grid_search_param(dataset_file_path, k_range,
                                         sigma_range,
                                         min_size_range,
                                         iou_threshold_range,
                                         verbose = True,
                                         visualize = False,
                                         save_to_file = False):
    number_combinations = len(k_range)*len(sigma_range)*len(min_size_range)*len(iou_threshold_range)
    combination = 1
    param_results = {}
    for k_ in k_range:
        for sigma_ in sigma_range:
            for min_size_ in min_size_range:
                for iou_threshold_ in iou_threshold_range:
                    print("Combination " + str(combination) + " out of " + str(number_combinations))
                    results = []
                    for i, file in enumerate(os.listdir(dataset_file_path)):
                        print("Image number " + str(i))
                        file_name = os.fsdecode(file)
                        file_path = dataset_file_path + "/" + file_name
                        PIL_image = Image.open(file_path)
                        guess = R_CNN(PIL_image, k_, sigma_, min_size_, iou_threshold_, verbose, visualize)
                        print(file_name[:-4], guess)
                        results.append([file_name[:-4], guess])
                        print("\n")
                    param_results[(k_,
                                  sigma_,
                                  min_size_,
                                  iou_threshold_)] = results
                    print("\n")
                    combination += 1
    if save_to_file: np.save('./result_metrics/grid_search_results.npy', param_results)
    return param_results

def results_to_score(param_results):
    param_scores = {}
    for param, results in param_results.items():
        normalized_scores = []
        for result in results:
            score = float(Levenshtein_dist(result[0], result[1]))
            normalized_scores.append(score/len(result[0]))
        param_scores[param] = np.mean(normalized_scores)
    return param_scores

def test_R_CNN(file_path, k, sigma, min_size, iou_threshold, verbose = True, visualize = True):
    # Open an image with PIL
    PIL_image = Image.open(file_path)

    # Feed it to the R_CNN algorithm
    license_plate_guess = R_CNN(PIL_image, k, sigma, min_size, iou_threshold, verbose, visualize)
    print("License plate guess: " + license_plate_guess)


test_R_CNN("./datasets/samples/license_plates_with_slight_context_sample/KLG1CA2555.png",
           3750, 0.8, 100, 0.15)
#results_ = grid_search_param("./datasets/samples/license_plates_with_slight_context_sample",
#                             [3250, 3500, 3750],
#                             [0.8],
#                             [100],
#                             [0.15],
#                             verbose = True,
#                             visualize = False,
#                             save_to_file = True)
#print(results_to_score(results_))