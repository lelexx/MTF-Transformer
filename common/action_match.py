import numpy as np
import json
action_match = {
        'S9':
            {'Directions': {'action_name': 'Directions', 'subaction_idx': 2, 'frames': 2699},
             'Directions 1': {'action_name': 'Directions', 'subaction_idx': 1, 'frames': 2356},
             'Discussion 1': {'action_name': 'Discussion', 'subaction_idx': 1, 'frames': 5873},
             'Discussion 2': {'action_name': 'Discussion', 'subaction_idx': 2, 'frames': 5306},
             'Eating': {'action_name': 'Eating', 'subaction_idx': 2, 'frames': 2686},
             'Eating 1': {'action_name': 'Eating', 'subaction_idx': 1, 'frames': 2663},
             'Greeting': {'action_name': 'Greeting', 'subaction_idx': 2, 'frames': 1447},
             'Greeting 1': {'action_name': 'Greeting', 'subaction_idx': 1, 'frames': 2711},
             'Phoning': {'action_name': 'Phoning', 'subaction_idx': 2, 'frames': 3319},
             'Phoning 1': {'action_name': 'Phoning', 'subaction_idx': 1, 'frames': 3821},
             'Photo': {'action_name': 'Photo', 'subaction_idx': 2, 'frames': 2346},
             'Photo 1': {'action_name': 'Photo', 'subaction_idx': 1, 'frames': 1449},
             'Posing': {'action_name': 'Posing', 'subaction_idx': 2, 'frames': 1964},
             'Posing 1': {'action_name': 'Posing', 'subaction_idx': 1, 'frames': 1968},
             'Purchases': {'action_name': 'Purchases', 'subaction_idx': 2, 'frames': 1529},
             'Purchases 1': {'action_name': 'Purchases', 'subaction_idx': 1, 'frames': 1226},
             'Sitting': {'action_name': 'Sitting', 'subaction_idx': 2, 'frames': 2962},
             'Sitting 1': {'action_name': 'Sitting', 'subaction_idx': 1, 'frames': 3071},
             'SittingDown': {'action_name': 'SittingDown', 'subaction_idx': 1, 'frames': 2932},
             'SittingDown 1': {'action_name': 'SittingDown', 'subaction_idx': 2, 'frames': 1554},
             'Smoking': {'action_name': 'Smoking', 'subaction_idx': 2, 'frames': 4334},
             'Smoking 1': {'action_name': 'Smoking', 'subaction_idx': 1, 'frames': 4377},
             'Waiting': {'action_name': 'Waiting', 'subaction_idx': 2, 'frames': 3312},
             'Waiting 1': {'action_name': 'Waiting', 'subaction_idx': 1, 'frames': 1612},
             'WalkDog': {'action_name': 'WalkDog', 'subaction_idx': 2, 'frames': 2237},
             'WalkDog 1': {'action_name': 'WalkDog', 'subaction_idx': 1, 'frames': 2217},
             'WalkTogether': {'action_name': 'WalkTogether', 'subaction_idx': 2, 'frames': 1703},
             'WalkTogether 1': {'action_name': 'WalkTogether', 'subaction_idx': 1, 'frames': 1685},
             'Walking': {'action_name': 'Walking', 'subaction_idx': 2, 'frames': 1612},
             'Walking 1': {'action_name': 'Walking', 'subaction_idx': 1, 'frames': 2446}},
        'S11': {'Directions 1': {'action_name': 'Directions', 'subaction_idx': 1, 'frames': 1552},
            'Discussion 1': {'action_name': 'Discussion', 'subaction_idx': 1, 'frames': 2684},
            'Discussion 2': {'action_name': 'Discussion', 'subaction_idx': 2, 'frames': 2198},
            'Eating': {'action_name': 'Eating', 'subaction_idx': 2, 'frames': 2203},
            'Eating 1': {'action_name': 'Eating', 'subaction_idx': 1, 'frames': 2275},
            'Greeting': {'action_name': 'Greeting', 'subaction_idx': 2, 'frames': 1808},
            'Greeting 2': {'action_name': 'Greeting', 'subaction_idx': 1, 'frames': 1695},
            'Phoning 2': {'action_name': 'Phoning', 'subaction_idx': 2, 'frames': 3492},
            'Phoning 3': {'action_name': 'Phoning', 'subaction_idx': 1, 'frames': 3390},
            'Photo': {'action_name': 'Photo', 'subaction_idx': 2, 'frames': 1990},
            'Photo 1': {'action_name': 'Photo', 'subaction_idx': 1, 'frames': 1545},
            'Posing': {'action_name': 'Posing', 'subaction_idx': 2, 'frames': 1407},
            'Posing 1': {'action_name': 'Posing', 'subaction_idx': 1, 'frames': 1481},
            'Purchases': {'action_name': 'Purchases', 'subaction_idx': 2, 'frames': 1040},
            'Purchases 1': {'action_name': 'Purchases', 'subaction_idx': 1, 'frames': 1026},
            'Sitting': {'action_name': 'Sitting', 'subaction_idx': 2, 'frames': 2179},
            'Sitting 1': {'action_name': 'Sitting', 'subaction_idx': 1, 'frames': 1857},
            'SittingDown': {'action_name': 'SittingDown', 'subaction_idx': 1, 'frames': 1841},
            'SittingDown 1': {'action_name': 'SittingDown', 'subaction_idx': 2, 'frames': 2004},
            'Smoking': {'action_name': 'Smoking', 'subaction_idx': 2, 'frames': 2410},
            'Smoking 2': {'action_name': 'Smoking', 'subaction_idx': 1, 'frames': 2767},
            'Waiting': {'action_name': 'Waiting', 'subaction_idx': 2, 'frames': 2262},
            'Waiting 1': {'action_name': 'Waiting', 'subaction_idx': 1, 'frames': 2280},
            'WalkDog': {'action_name': 'WalkDog', 'subaction_idx': 2, 'frames': 1435},
            'WalkDog 1': {'action_name': 'WalkDog', 'subaction_idx': 1, 'frames': 1187},
            'WalkTogether': {'action_name': 'WalkTogether', 'subaction_idx': 2, 'frames': 1360},
            'WalkTogether 1': {'action_name': 'WalkTogether', 'subaction_idx': 1, 'frames': 1793},
            'Walking': {'action_name': 'Walking', 'subaction_idx': 2, 'frames': 1621},
            'Walking 1': {'action_name': 'Walking', 'subaction_idx': 1, 'frames': 1637}}

        }
def get_action_name(action_name, num_frames):
    for k, v in action_match['S9'].items():
        if v['action_name'] == action_name and v['frames'] == num_frames:
            return v['subaction_idx']

def get_num_frames(subject, action):
    return action_match['S{}'.format(subject)][action]['frames']