import numpy as np
import pickle
from plyfile import PlyData


JOINT_TO_PART = [
    4,  # 0: Pelvis
    1,  # 1: Left hip
    1,  # 2: Right hip
    0,  # 3: Spine1
    1,  # 4: Left knee
    1,  # 5: Right knee
    0,  # 6: Spine2
    1,  # 7: Left ankle
    1,  # 8: Right ankle
    0,  # 9: Spine3
    1,  # 10: Left foot
    1,  # 11: Right foot
    3,  # 12: Neck
    2,  # 13: Left collar
    2,  # 14: Right collar
    3,  # 15: Head
    2,  # 16: Left shoulder
    2,  # 17: Right shoulder
    2,  # 18: Left elbow
    2,  # 19: Right elbow
    2,  # 20: Left wrist
    2,  # 21: Right wrist
    2,  # 22: Left hand
    2,  # 23: Right hand
]

PART_COLOR = [
    [226, 226, 226],
    [129, 0, 50],
    [243, 115, 68],
    [228, 162, 227],
    [210, 78, 142],
    [152, 78, 163],
]

# import numpy as np
# import pickle
# from plyfile import PlyData
#
# SMPL_JOINT_NAMES = [
#     "pelvis",
#     "left_hip",
#     "right_hip",
#     "spine1",
#     "left_knee",
#     "right_knee",
#     "spine2",
#     "left_ankle",
#     "right_ankle",
#     "spine3",
#     "left_foot",
#     "right_foot",
#     "neck",
#     "left_collar",
#     "right_collar",
#     "head",
#     "left_shoulder",
#     "right_shoulder",
#     "left_elbow",
#     "right_elbow",
#     "left_wrist",
#     "right_wrist",
#     "left_hand",
#     "right_hand",
# ]
#
# PART_NAMES=[
#     "background",
#     "rightHand",
#     "rightUpLeg",
#     "leftArm",
#     "head",
#     "leftLeg",
#     "leftFoot",
#     "torso",
#     "rightFoot",
#     "rightArm",
#     "leftHand",
#     "rightLeg",
#     "leftForeArm",
#     "rightForeArm",
#     "leftUpLeg",
#     "hips",
# ]
#
# JOINT_TO_PART=[
#     15,
#     14,2,
#     15,
#     5,11,
#     7,
#     6,8,
#     7,
#     6,8,
#     4,
#     7,7,
#     4,
#     3,9,        #shoulder
#     12,13,      #elbow'
#     1,10,       #wrist
#     1,10,       #hand
# ]
#
# PART_COLOR=[
#             [226, 226, 226],
#             [158, 143, 143],  # rightHand
#             [243, 115, 68],  # rightUpLeg
#             [228, 162, 227], # leftArm
#             [210, 78, 142],  # head
#             [152, 78, 163],  # leftLeg
#             [76 , 134, 26],  # leftFoot
#             [100, 143, 255], # torso
#             [129, 0  , 50],  # rightFoot
#             [255, 176, 0],   # rightArm
#             [192, 100, 119], # leftHand
#             [149, 192, 228], # rightLeg
#             [243, 232, 88],  # leftForeArm
#             [90 , 64 , 210], # rightForeArm
#             [152, 200, 156], # leftUpLeg
#             [129, 103, 106], # hips
#             ]

def generate_ply(points, colors, labels):
    arr = [(points[i][0], points[i][1], points[i][2],
            colors[i][0], colors[i][1], colors[i][2],
            labels[i]) for i in range(len(points))]

    arr = np.array(arr, dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'uint8'),
        ('green', 'uint8'),
        ('blue', 'uint8'),
        ('label', 'uint8')])

    from plyfile import PlyElement
    el = PlyElement.describe(arr, 'vertex')
    plydata = PlyData([el], False, '<')

    return plydata

def save_plyfile(plydata: PlyData, file_path):
    print(f'save to {file_path}')
    with open(file_path, "wb") as f:
        plydata.text = False
        plydata.write(f)

def get_colors_from_labels(labels):
    colors = []
    for label in labels:
        colors.append(PART_COLOR[label])
    return colors

if __name__ == '__main__':
    import pickle
    import numpy as np
    from plyfile import PlyData

    smpl_path = './body_models/smpl/neutral/model.pkl'
    save_path = './body_models/smpl/neutral/smpl_semantic_sim.ply'
    with open(smpl_path, 'rb') as f:
        params = pickle.load(f, encoding="latin1")
        smpl_weights = params['weights']
        smpl_v_template = params['v_template']

    labels = []
    for w in smpl_weights:
        labels.append(JOINT_TO_PART[np.argmax(w)])

    colors = get_colors_from_labels(labels)

    ply = generate_ply(smpl_v_template, colors, labels)
    save_plyfile(ply, save_path)