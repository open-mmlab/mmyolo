# Copyright (c) OpenMMLab. All rights reserved.
# Reference: https://github.com/jbwang1997/BboxToolkit

import argparse
import codecs
import datetime
import itertools
import os
import os.path as osp
import time
from functools import partial, reduce
from math import ceil
from multiprocessing import Manager, Pool
from typing import List, Sequence

import cv2
import numpy as np
from mmengine import Config, MMLogger, mkdir_or_exist, print_log
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    import shapely.geometry as shgeo
except ImportError:
    raise ImportError('Please run "pip install shapely" '
                      'to install shapely first.')

PHASE_REQUIRE_SETS = dict(
    trainval=['train', 'val'],
    train=[
        'train',
    ],
    val=[
        'val',
    ],
    test=[
        'test',
    ],
)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'split_config', type=str, help='The split config for image slicing.')
    parser.add_argument(
        'data_root', type=str, help='Root dir of DOTA dataset.')
    parser.add_argument(
        'out_dir', type=str, help='Output dir for split result.')
    parser.add_argument(
        '--ann-subdir',
        default='labelTxt-v1.0',
        type=str,
        help='output directory')
    parser.add_argument(
        '--phase',
        '-p',
        nargs='+',
        default=['trainval', 'test'],
        type=str,
        choices=['trainval', 'train', 'val', 'test'],
        help='Phase of the data set to be prepared.')
    parser.add_argument(
        '--nproc', default=8, type=int, help='Number of processes.')
    parser.add_argument(
        '--save-ext',
        default=None,
        type=str,
        help='Extension of the saved image.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to allow overwrite if annotation folder exist.')
    args = parser.parse_args()

    assert args.split_config is not None, "argument split_config can't be None"
    split_cfg = Config.fromfile(args.split_config)

    # assert arguments
    assert args.data_root is not None, "argument data_root can't be None"
    if args.save_ext:
        assert args.save_ext in ['png', 'jpg', 'bmp', 'tif']

    assert len(split_cfg.patch_sizes) == len(split_cfg.patch_overlap_sizes)
    assert 0 <= split_cfg.iof_thr <= 1
    if split_cfg.get('padding'):
        padding_value = split_cfg.get('padding_value')
        assert padding_value is not None, \
            "padding_value can't be None when padding is True."
        padding_value = padding_value[0] \
            if len(padding_value) == 1 else padding_value
        split_cfg.padding_value = padding_value
    else:
        split_cfg.padding = False
        split_cfg.padding_value = None
    return args, split_cfg


def _make_dirs(out_dir: str, phase: List[str], allow_overwrite: bool):
    """Prepare folder for DOTA dataset.

    Args:
        out_dir (str): The output dir for DOTA split.
        phase (List[str]): The phase to prepare.
        allow_overwrite (bool): Whether to allow overwrite when folder exist.
    """
    logger = MMLogger.get_current_instance()
    for p in phase:
        phase_dir = osp.join(out_dir, p)
        if not allow_overwrite:
            assert not osp.exists(phase_dir), \
                f'{osp.join(phase_dir)} already exists,' \
                'If you want to ignore existing files, set --overwrite'
        else:
            if osp.exists(phase_dir):
                logger.warning(
                    f'{p} set in {osp.join(phase_dir)} will be overwritten')
        mkdir_or_exist(phase_dir)
        mkdir_or_exist(osp.join(phase_dir, 'images'))
        mkdir_or_exist(osp.join(phase_dir, 'annfiles'))


def load_original_annotations(data_root: str,
                              ann_subdir: str = 'labelTxt-v1.0',
                              phase: str = 'train',
                              nproc: int = 8):
    img_dir = osp.join(data_root, phase, 'images')
    assert osp.isdir(img_dir), f'The {img_dir} is not an existing dir!'

    if phase == 'test':
        ann_dir = None
    else:
        ann_dir = osp.join(data_root, phase, ann_subdir, 'labelTxt')
        assert osp.isdir(ann_dir), f'The {ann_dir} is not an existing dir!'

    _load_func = partial(_load_dota_single, img_dir=img_dir, ann_dir=ann_dir)
    if nproc > 1:
        pool = Pool(nproc)
        contents = pool.map(_load_func, os.listdir(img_dir))
        pool.close()
    else:
        contents = list(map(_load_func, os.listdir(img_dir)))
    infos = [c for c in contents if c is not None]
    return infos


def _load_dota_single(imgfile: str, img_dir: str, ann_dir: str):
    """Load DOTA's single image.

    Args:
        imgfile (str): Filename of single image.
        img_dir (str): Path of images.
        ann_dir (str): Path of annotations.

    Returns:
        result (dict): Information of a single image.

        - ``id``: Image id.
        - ``filename``: Filename of single image.
        - ``filepath``: Filepath of single image.
        - ``width``: The width of image.
        - ``height``: The height of image.
        - ``annotations``: The annotation of single image.
        - ``gsd``: The ground sampling distance.
    """
    img_id, ext = osp.splitext(imgfile)
    if ext not in ['.jpg', '.JPG', '.png', '.tif', '.bmp']:
        return None

    imgpath = osp.join(img_dir, imgfile)
    size = Image.open(imgpath).size
    txtfile = None if ann_dir is None else osp.join(ann_dir, img_id + '.txt')
    content = _load_dota_txt(txtfile)

    content.update(
        dict(
            width=size[0],
            height=size[1],
            filename=imgfile,
            filepath=imgpath,
            id=img_id))
    return content


def _load_dota_txt(txtfile):
    """Load DOTA's txt annotation.

    Args:
        txtfile (str): Filename of single Dota txt annotation.

    Returns:
        result (dict): Annotation of single image.

        - ``annotations``: The annotation of single image.
        - ``gsd``: The ground sampling distance.
    """
    gsd, bboxes, labels, diffs = None, [], [], []
    if txtfile is None:
        pass
    elif not osp.isfile(txtfile):
        print(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile) as f:
            for line in f:
                if line.startswith('gsd'):
                    num = line.split(':')[-1]
                    try:
                        gsd = float(num)
                    except ValueError:
                        gsd = None
                    continue

                items = line.split(' ')
                if len(items) >= 9:
                    bboxes.append([float(i) for i in items[:8]])
                    labels.append(items[8])
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
        np.zeros((0, 8), dtype=np.float32)
    diffs = np.array(diffs, dtype=np.int64) if diffs else \
        np.zeros((0,), dtype=np.int64)
    ann = dict(bboxes=bboxes, labels=labels, diffs=diffs)
    return dict(gsd=gsd, annotations=ann)


def poly2hbb(polys):
    """Convert polygons to horizontal bboxes.

    Args:
        polys (np.array): Polygons with shape (N, 8)

    Returns:
        np.array: Horizontal bboxes.
    """
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)


def get_sliding_window(info, patch_settings, img_rate_thr):
    """Get sliding windows.

    Args:
        info (dict): Dict of image's width and height.
        patch_settings (list): List of patch settings,
            each in format (patch_size, patch_overlap).
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        list[np.array]: Information of valid windows.
    """
    eps = 0.01
    windows = []
    width, height = info['width'], info['height']
    for (size, gap) in patch_settings:
        assert size > gap, f'invaild size gap pair [{size} {gap}]'
        step = size - gap

        x_num = 1 if width <= size else ceil((width - size) / step + 1)
        x_start = [step * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size > width:
            x_start[-1] = width - size

        y_num = 1 if height <= size else ceil((height - size) / step + 1)
        y_start = [step * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size > height:
            y_start[-1] = height - size

        start = np.array(
            list(itertools.product(x_start, y_start)), dtype=np.int64)
        stop = start + size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates > img_rate_thr).any():
        max_rate = img_rates.max()
        img_rates[abs(img_rates - max_rate) < eps] = 1
    return windows[img_rates > img_rate_thr]


def get_window_annotation(info, windows, iof_thr):
    """Get annotation by sliding windows.

    Args:
        info (dict): Dict of bbox annotations.
        windows (np.array): information of sliding windows.
        iof_thr (float): Threshold of overlaps between bbox and window.

    Returns:
        list[dict]: List of bbox annotations of every window.
    """
    bboxes = info['annotations']['bboxes']
    iofs = ann_window_iof(bboxes, windows)

    window_anns = []
    for i in range(windows.shape[0]):
        win_iofs = iofs[:, i]
        pos_inds = np.nonzero(win_iofs >= iof_thr)[0].tolist()

        win_ann = dict()
        for k, v in info['annotations'].items():
            try:
                win_ann[k] = v[pos_inds]
            except TypeError:
                win_ann[k] = [v[i] for i in pos_inds]
        win_ann['trunc'] = win_iofs[pos_inds] < 1
        window_anns.append(win_ann)
    return window_anns


def ann_window_iof(anns, window, eps=1e-6):
    """Compute overlaps (iof) between annotations (poly) and window (hbox).

    Args:
        anns (np.array): quadri annotations with shape (n, 8).
        window (np.array): slide windows with shape (m, 4).
        eps (float, optional): Defaults to 1e-6.

    Returns:
        np.array: iof between box and window.
    """
    rows = anns.shape[0]
    cols = window.shape[0]

    if rows * cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)

    hbboxes_ann = poly2hbb(anns)
    hbboxes_win = window
    hbboxes_ann = hbboxes_ann[:, None, :]
    lt = np.maximum(hbboxes_ann[..., :2], hbboxes_win[..., :2])
    rb = np.minimum(hbboxes_ann[..., 2:], hbboxes_win[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    l, t, r, b = (window[..., i] for i in range(4))
    polys_win = np.stack([l, t, r, t, r, b, l, b], axis=-1)
    sg_polys_ann = [shgeo.Polygon(p) for p in anns.reshape(rows, -1, 2)]
    sg_polys_win = [shgeo.Polygon(p) for p in polys_win.reshape(cols, -1, 2)]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys_ann[p[0]].intersection(sg_polys_win[p[-1]]).area
    unions = np.array([p.area for p in sg_polys_ann], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def crop_and_save_img(info, windows, window_anns, padding, padding_value,
                      save_dir, anno_dir, img_ext):
    """Crop the image and save.

    Args:
        info (dict): Image's information.
        windows (np.array): information of sliding windows.
        window_anns (list[dict]): List of bbox annotations of every window.
        padding (bool): If True, with padding.
        padding_value (tuple[int|float]): Padding value.
        save_dir (str): Save filename.
        anno_dir (str): Annotation filename.
        img_ext (str): Picture suffix.

    Returns:
        list[dict]: Information of paths.
    """
    img = cv2.imread(info['filepath'])
    patch_infos = []
    for window, ann in zip(windows, window_anns):
        patch_info = dict()
        for k, v in info.items():
            if k not in [
                    'id', 'filename', 'filepath', 'width', 'height',
                    'annotations'
            ]:
                patch_info[k] = v

        x_start, y_start, x_stop, y_stop = window.tolist()
        patch_info['x_start'] = x_start
        patch_info['y_start'] = y_start
        patch_info['id'] = \
            info['id'] + '__' + str(x_stop - x_start) + \
            '__' + str(x_start) + '___' + str(y_start)
        patch_info['ori_id'] = info['id']

        ann['bboxes'] = shift_qbboxes(ann['bboxes'], [-x_start, -y_start])
        patch_info['ann'] = ann

        patch = img[y_start:y_stop, x_start:x_stop]
        if padding:
            height = y_stop - y_start
            width = x_stop - x_start
            if height > patch.shape[0] or width > patch.shape[1]:
                padding_patch = np.empty((height, width, patch.shape[-1]),
                                         dtype=np.uint8)
                if not isinstance(padding_value, (int, float)):
                    assert len(padding_value) == patch.shape[-1]
                padding_patch[...] = padding_value
                padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                patch = padding_patch
        patch_info['height'] = patch.shape[0]
        patch_info['width'] = patch.shape[1]

        cv2.imwrite(
            osp.join(save_dir, patch_info['id'] + '.' + img_ext), patch)
        patch_info['filename'] = patch_info['id'] + '.' + img_ext
        patch_infos.append(patch_info)

        bboxes_num = patch_info['ann']['bboxes'].shape[0]
        outdir = os.path.join(anno_dir, patch_info['id'] + '.txt')

        with codecs.open(outdir, 'w', 'utf-8') as f_out:
            if bboxes_num == 0:
                pass
            else:
                for idx in range(bboxes_num):
                    obj = patch_info['ann']
                    outline = ' '.join(list(map(str, obj['bboxes'][idx])))
                    diffs = str(
                        obj['diffs'][idx]) if not obj['trunc'][idx] else '2'
                    outline = outline + ' ' + obj['labels'][idx] + ' ' + diffs
                    f_out.write(outline + '\n')

    return patch_infos


def shift_qbboxes(bboxes, offset: Sequence[float]):
    """Map bboxes from window coordinate back to original coordinate. TODO
    Refactor and move to `mmyolo/utils/large_image.py`

    Args:
        bboxes (np.array): quadrilateral boxes with window coordinate.
        offset (Sequence[float]): The translation offsets with shape of (2, ).

    Returns:
        np.array: bboxes with original coordinate.
    """
    dim = bboxes.shape[-1]
    translated = bboxes + np.array(offset * int(dim / 2), dtype=np.float32)
    return translated


def single_split(info, patch_settings, min_img_ratio, iof_thr, padding,
                 padding_value, save_dir, anno_dir, img_ext, lock, prog,
                 total):
    """Single image split. TODO Refactoring to make it more generic.

    Args:
        info (dict): Image info and annotations.
        patch_settings (list): List of patch settings,
            each in format (patch_size, patch_overlap).
        min_img_ratio (float): Threshold of window area divided by image area.
        iof_thr (float): Threshold of overlaps between bbox and window.
        padding (bool): If True, with padding.
        padding_value (tuple[int|float]): Padding value.
        save_dir (str): Save filename.
        anno_dir (str): Annotation filename.
        img_ext (str): Picture suffix.
        lock (Lock): Lock of Manager.
        prog (object): Progress of Manager.
        total (int): Length of infos.

    Returns:
        list[dict]: Information of paths.
    """
    img_ext = img_ext if img_ext is not None else info['filename'].split(
        '.')[-1]
    windows = get_sliding_window(info, patch_settings, min_img_ratio)
    window_anns = get_window_annotation(info, windows, iof_thr)
    patch_infos = crop_and_save_img(info, windows, window_anns, padding,
                                    padding_value, save_dir, anno_dir, img_ext)
    assert patch_infos

    lock.acquire()
    prog.value += 1
    msg = f'({prog.value / total:3.1%} {prog.value}:{total})'
    msg += ' - ' + f"Filename: {info['filename']}"
    msg += ' - ' + f"width: {info['width']:<5d}"
    msg += ' - ' + f"height: {info['height']:<5d}"
    msg += ' - ' + f"Objects: {len(info['annotations']['bboxes']):<5d}"
    msg += ' - ' + f'Patches: {len(patch_infos)}'
    print_log(msg, 'current')
    lock.release()

    return patch_infos


def main():
    args, split_cfg = parse_args()

    mkdir_or_exist(args.out_dir)

    # init logger
    log_file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    logger: MMLogger = MMLogger.get_instance(
        'mmyolo',
        log_file=osp.join(args.out_dir, log_file_name),
        log_level='INFO')

    # print configs
    arg_str = ''
    for arg in args._get_kwargs():
        arg_str += arg[0] + ' = ' + str(arg[1]) + '\n'

    logger.info('Base Settings:\n' + arg_str)
    logger.info('Split Settings:\n' + split_cfg.pretty_text)

    # make dirs
    _make_dirs(args.out_dir, args.phase, args.overwrite)

    # Load original dota data
    required_sets = []
    for p in args.phase:
        required_sets.extend(PHASE_REQUIRE_SETS[p])
    required_sets = set(required_sets)

    loaded_data_set = dict()
    for req_set in required_sets:
        logger.info(f'Starting loading DOTA {req_set} set information.')
        start_time = time.time()

        infos = load_original_annotations(
            data_root=args.data_root,
            ann_subdir=args.ann_subdir,
            phase=req_set)

        end_time = time.time()
        result_log = f'Finishing loading {req_set} set, '
        result_log += f'get {len(infos)} images, '
        result_log += f'using {end_time - start_time:.3f}s.'
        logger.info(result_log)

        loaded_data_set[req_set] = infos

    # Preprocess patch settings
    patch_settings = []
    for ratio in split_cfg.img_resize_ratio:
        for size, gap in zip(split_cfg.patch_sizes,
                             split_cfg.patch_overlap_sizes):
            size_gap = (int(size / ratio), int(gap / ratio))
            if size_gap not in patch_settings:
                patch_settings.append(size_gap)

    # Split data
    for p in args.phase:
        save_imgs_dir = osp.join(args.out_dir, p, 'images')
        save_anns_dir = osp.join(args.out_dir, p, 'annfiles')

        logger.info(f'Start splitting {p} set images!')
        start = time.time()
        manager = Manager()

        data_infos = []
        for req_set in PHASE_REQUIRE_SETS[p]:
            data_infos.extend(loaded_data_set[req_set])

        worker = partial(
            single_split,
            patch_settings=patch_settings,
            min_img_ratio=split_cfg.min_img_ratio,
            iof_thr=split_cfg.iof_thr,
            padding=split_cfg.padding,
            padding_value=split_cfg.padding_value,
            save_dir=save_imgs_dir,
            anno_dir=save_anns_dir,
            img_ext=args.save_ext,
            lock=manager.Lock(),
            prog=manager.Value('i', 0),
            total=len(data_infos))

        if args.nproc > 1:
            pool = Pool(args.nproc)
            patch_infos = pool.map(worker, data_infos)
            pool.close()
        else:
            patch_infos = list(map(worker, data_infos))

        patch_infos = reduce(lambda x, y: x + y, patch_infos)
        stop = time.time()
        logger.info(
            f'Finish splitting {p} set images in {int(stop - start)} second!!!'
        )
        logger.info(f'Total images number: {len(patch_infos)}')


if __name__ == '__main__':
    main()
