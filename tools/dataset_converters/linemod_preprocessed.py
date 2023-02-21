import argparse
import os

import cv2
import mmcv
import mmengine
import numpy as np


def parse_examples(data_file):
    if not os.path.isfile(data_file):
        print(f'Error: file {data_file} does not exist!')
        return None

    with open(data_file) as fid:
        data_examples = [example.strip() for example in fid if example != '']

    return data_examples


def insert_np_cam_calibration(filtered_infos):
    for info in filtered_infos:
        info['cam_K_np'] = np.reshape(np.array(info['cam_K']), newshape=(3, 3))

    return filtered_infos


def get_bbox_from_mask(mask, mask_value=None):
    if mask_value is None:
        seg = np.where(mask != 0)
    else:
        seg = np.where(mask == mask_value)
    # check if mask is empty
    if seg[0].size <= 0 or seg[1].size <= 0:
        return np.zeros((4, ), dtype=np.float32), False
    min_x = np.min(seg[1])
    min_y = np.min(seg[0])
    max_x = np.max(seg[1])
    max_y = np.max(seg[0])

    return np.array([min_x, min_y, max_x, max_y], dtype=np.float32), True


def project_points_3D_to_2D(points_3D, rotation_vector, translation_vector,
                            camera_matrix):
    rotation_vector = np.reshape(rotation_vector, newshape=(3, 3))
    points_2D, _ = cv2.projectPoints(points_3D, rotation_vector,
                                     translation_vector, camera_matrix, None)
    points_2D = np.squeeze(points_2D)

    return points_2D


def convert_gt(id, gt_list, info_list, mask_paths, model_info_dict):
    instances = []
    rotation_parameter = 9
    translation_parameter = 3
    for gt, info, mask_path in zip(gt_list, info_list, mask_paths):
        # init annotations in the correct base format.
        # set number of annotations to one
        # because linemod dataset only contains one annotation per image

        # +1 for is_symmetric flag
        num_all_rotation_parameters = rotation_parameter + 1

        instance = {
            'bbox_label': str,
            'bbox': np.zeros((1, 4)),
            'rotation': np.zeros((1, num_all_rotation_parameters)),
            'translation': np.zeros((1, translation_parameter)),
            'center': np.zeros((1, 2)),
            'corners': dict,
        }

        # fill in the values
        # get bbox from mask
        instance['bbox_label'] = id
        mask = cv2.imread(mask_path)
        instance['bbox'][0, :], _ = get_bbox_from_mask(mask)
        instance['rotation'][0, :-1] = np.array(gt['cam_R_m2c'])
        instance['rotation'][0, -1] = float(0)
        instance['translation'][0, :] = np.array(gt['cam_t_m2c'])
        instance['center'] = project_points_3D_to_2D(
            points_3D=np.zeros(shape=(1, 3)),
            # transform the object origin point which is the centerpoint
            rotation_vector=np.array(gt['cam_R_m2c']),
            translation_vector=np.array(gt['cam_t_m2c']),
            camera_matrix=info['cam_K_np'])

        min_x = model_info_dict[int(id)]['min_x']
        min_y = model_info_dict[int(id)]['min_y']
        min_z = model_info_dict[int(id)]['min_z']
        max_x = min_x + model_info_dict[int(id)]['size_x']
        max_y = min_y + model_info_dict[int(id)]['size_y']
        max_z = min_z + model_info_dict[int(id)]['size_z']
        instance['corners'] = np.array([[min_x, min_y, min_z],
                                        [min_x, min_y, max_z],
                                        [min_x, max_y, min_z],
                                        [min_x, max_y, max_z],
                                        [max_x, min_y, min_z],
                                        [max_x, min_y, max_z],
                                        [max_x, max_y, min_z],
                                        [max_x, max_y, max_z]])

        instance['corners'] = [
            project_points_3D_to_2D(corner, np.array(gt['cam_R_m2c']),
                                    np.array(gt['cam_t_m2c']),
                                    info['cam_K_np'])
            for corner in instance['corners']
        ]

        instances.append(instance)

    return instances


def convert_linemod(id, object_path, data_examples, gt_dict, info_dict,
                    model_info_dict):
    all_images_path = os.path.join(object_path, 'rgb')
    all_filenames = [
        filename for filename in os.listdir(all_images_path)
        if '.png' in filename and filename.replace('.png', '') in data_examples
    ]
    image_paths = [
        os.path.join(all_images_path, filename) for filename in all_filenames
    ]
    mask_paths = [
        image_path.replace('rgb', 'mask') for image_path in image_paths
    ]
    depth_paths = [
        image_path.replace('rgb', 'depth') for image_path in image_paths
    ]

    example_ids = [int(filename.split('.')[0]) for filename in all_filenames]
    filtered_gt_lists = [gt_dict[key] for key in example_ids]
    filtered_gts = []
    for gt_list in filtered_gt_lists:
        all_annos = [anno for anno in gt_list if anno['obj_id'] == int(id)]
        if len(all_annos) <= 0:
            print('\nError: No annotation found!')
            filtered_gts.append(None)
        elif len(all_annos) > 1:
            print('\nWarning: found more than one annotation.\
                    using only the first annotation')
            filtered_gts.append(all_annos[0])
        else:
            filtered_gts.append(all_annos[0])

    filtered_infos = [info_dict[key] for key in example_ids]
    infos = insert_np_cam_calibration(filtered_infos)
    instances = convert_gt(args.id, filtered_gts, infos, mask_paths,
                           model_info_dict)
    data_list = []
    for i, id in enumerate(all_filenames):
        filename = all_filenames[i]
        img_path = os.path.join(all_images_path, filename)
        image = mmcv.imread(img_path)
        height, width = image.shape[:2]
        image_id = example_ids[i]
        instances_list = instances[i]

        data_info = dict(
            id=f'{image_id}.png',
            img_path=image_paths[i],
            mask_path=mask_paths[i],
            depth_path=depth_paths[i],
            width=width,
            height=height,
            cam_K=infos[i]['cam_K_np'],
            depth_scale=infos[i]['depth_scale'],
            instances=[instances_list])

        # data_list.update(data_info)
        data_list.append(data_info)
    return data_list


def parse_args():
    parser = argparse.ArgumentParser(description='Create_linemod_json')
    parser.add_argument('--root', help='root path')
    parser.add_argument('--id', type=str, help='object id, for example: 01')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    object_path = os.path.join(args.root, f'data/{args.id}/')
    model_info_path = args.root + 'models/'
    data_examples = parse_examples(object_path + 'train.txt')
    gt_dict = mmengine.load(object_path + 'gt.yml')
    info_dict = mmengine.load(object_path + 'info.yml')
    model_info_dict = mmengine.load(model_info_path + 'models_info.yml')

    metainfo = dict(
        dataset_type='linemod_preprocessed',
        taskname='PoseEstimation',
        classes={
            'ape', 'benchvise', 'bowl', 'cam', 'can', 'cat', 'cup', 'driller',
            'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone'
        })

    data_list = convert_linemod(args.id, object_path, data_examples, gt_dict,
                                info_dict, model_info_dict)

    model_path_list = dict(
        ape='',
        benchvise='',
        bowl='',
        cam='',
        can='',
        cat='',
        cup='',
        driller='',
        duck='',
        eggbox='',
        glue='',
        holepuncher='',
        iron='',
        lamp='',
        phone='',
    )
    for i, name in enumerate(model_path_list):
        if i < 10:
            model_path = model_info_path + 'obj_0' + f'{i}' + '.ply'
        else:
            model_path = model_info_path + 'obj_' + f'{i}' + '.ply'
        model_path_list[name] = model_path
    model_list = dict(model_path_list=model_path_list)

    out = dict(metainfo=metainfo, data_list=data_list, model_list=model_list)
    out_file = args.root + 'json/linemod_preprocessed_train.json'

    mmengine.dump(out, out_file)
