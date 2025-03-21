import numpy as np


### Measures

def inter_over_union(bbox_1, bbox_2):
    bbox_inter = [max(bbox_1[0], bbox_2[0]),
                  max(bbox_1[1], bbox_2[1]),
                  min(bbox_1[2], bbox_2[2]),
                  min(bbox_1[3], bbox_2[3])]
    area_1 = (bbox_1[2] - bbox_1[0])*(bbox_1[3] - bbox_1[1])
    area_2 = (bbox_2[2] - bbox_2[0])*(bbox_2[3] - bbox_2[1])
    area_inter = max(0, (bbox_inter[2] - bbox_inter[0]))*max(0, (bbox_inter[3] - bbox_inter[1]))
    return area_inter/(area_1 + area_2 - area_inter)

def one_in_other(bbox_1, bbox_2):
    first_in_second = (bbox_2[0] <= bbox_1[0]
                       and bbox_2[1] <= bbox_1[1]
                       and bbox_2[2] >= bbox_1[2]
                       and bbox_2[3] >= bbox_1[3])
    second_in_first = (bbox_1[0] <= bbox_2[0]
                       and bbox_1[1] <= bbox_2[1]
                       and bbox_1[2] >= bbox_2[2]
                       and bbox_1[3] >= bbox_2[3])
    return first_in_second or second_in_first


### Non-maximum suppression algorithm

def sort_matches_bboxes(matches, bboxes):
    matches = [matches[i] for i in np.argsort([match[2] for match in matches])]
    bboxes = [bboxes[match[0]] for match in matches]
    return matches, bboxes

def NMS(matches, bboxes, iou_threshold = 0.5):
    kept_matches = []
    kept_bboxes = []
    matches, bboxes = sort_matches_bboxes(matches, bboxes)
    while len(matches) > 0:
        match = matches.pop()
        bbox = bboxes.pop()
        kept_matches.append(match)
        kept_bboxes.append(bbox)

        i = 0
        number_matches = len(matches)
        while i < number_matches:
            #if match[1] == other_match[1]:
            if inter_over_union(bbox, bboxes[i]) > iou_threshold \
                    or one_in_other(bbox, bboxes[i]):
                matches.pop(i)
                bboxes.pop(i)
            else:
                i += 1
            number_matches = len(matches)
    return kept_matches, kept_bboxes
