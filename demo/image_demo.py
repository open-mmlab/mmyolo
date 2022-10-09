# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules

import os
import glob
from tqdm import tqdm
import cv2

IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp']
VIDEO_EXTENSIONS=('avi','rmvb','mkv','asf','wmv','mp4','3gp','flv')

def parse_args():
    parser = ArgumentParser()
    #img,video,camrea
    parser.add_argument('--mode',default='img',help='inference mode,img,video or camrea')
    parser.add_argument('--config', default='../configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py',help='Config file')
    parser.add_argument('--checkpoint',default='../weights/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth', help='Checkpoint file')
    parser.add_argument('--data-path', default='.//imgs', help='data file/dir')
    parser.add_argument('--out', default='result_imgs', help='Path to output file/dir')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument('--inference_subdir',action='store_false',help='inference image under subdirectories')
    parser.add_argument('--wait_time',type=int,default=1,help='Waiting time')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    parser.add_argument('--show', action='store_true', help='Show result')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def check_args(args):
    mode=args.mode
    if mode=='img':
        if os.path.isdir(args.data_path):
            return

        data_extenstion=args.data_path.split('.')[-1]
        assert data_extenstion in IMG_EXTENSIONS,'Incorrect file suffix'
        out_extenstion = args.out.split('.')[-1]
        assert out_extenstion in IMG_EXTENSIONS, 'Incorrect file suffix'

    elif mode=='video':
        data_extenstion = args.data_path.split('.')[-1]
        assert data_extenstion in VIDEO_EXTENSIONS, 'Incorrect file suffix'
        out_extenstion = args.out.split('.')[-1]
        assert out_extenstion in VIDEO_EXTENSIONS, 'Incorrect file suffix'

    elif mode=='camera':
        camera_id=args.camera_id
        print('use camera：'+str(camera_id))

    else:
        raise RuntimeError('Wrong mode')

def inference_image(args,model,visualizer):
    if os.path.isdir(args.data_path):
        if args.out:
            if not os.path.exists(args.out):
                os.mkdir(args.out)

        if args.inference_subdir:
            img_list = []
            img_out = []
            for root_dir, dirs, files in os.walk(args.data_path):
                for file in files:
                    if file.split('.')[-1] in IMG_EXTENSIONS:
                        img_path = os.path.join(root_dir, file)
                        img_list.append(img_path)
                        img_path = img_path.replace('//', '/')
                        img_path = img_path.replace('\\', '/')
                        path_split = img_path.split('/')
                        frist_dir = str(path_split[0])
                        #if use './' or '../',remove it
                        frist_dir = frist_dir.replace('.', '')
                        path_split[0] = frist_dir
                        out_path = os.path.join(args.out, '_'.join(path_split))
                        print(out_path)
                        img_out.append(out_path)

            for i, img_path in enumerate(tqdm(img_list)):
                img = mmcv.imread(img_path)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                out_path = ''
                if args.out:
                    out_path = img_out[i]
                result = inference_detector(model, img)
                visualizer.add_datasample(
                    'result',
                    img,
                    data_sample=result,
                    draw_gt=False,
                    show=args.show,
                    wait_time=args.wait_time,
                    out_file=out_path,
                    pred_score_thr=args.score_thr)

        else:
            img_list = []
            for suffix in IMG_EXTENSIONS:
                img_list += glob.glob(os.path.join(args.data_path, "*" + suffix))

            for img_path in tqdm(img_list):
                img = mmcv.imread(img_path)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                out_path = ''
                if args.out:
                    out_path = os.path.join(args.out, os.path.basename(img_path))
                result = inference_detector(model, img)
                visualizer.add_datasample(
                    'result',
                    img,
                    data_sample=result,
                    draw_gt=False,
                    show=args.show,
                    wait_time=args.wait_time,
                    out_file=out_path,
                    pred_score_thr=args.score_thr)

    else:
        img = mmcv.imread(args.data_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        result = inference_detector(model, img)
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=args.wait_time,
            out_file=args.out,
            pred_score_thr=args.score_thr)

def inference_video(args,model,visualizer):
    video_reader = mmcv.VideoReader(args.data_path)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in tqdm(video_reader):
        result = inference_detector(model, frame)
        visualizer.add_datasample(
            'result',
            frame,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr)
        frame=visualizer.get_image()
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def inference_camera(args,model,visualizer):
    camera = cv2.VideoCapture(args.camera_id)

    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, 20.0,(640,480))
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=args.score_thr)
        img = visualizer.get_image()
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(img, 'video', args.wait_time)
        if args.out:
            video_writer.write(img)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def main(args):
    check_args(args)

    # register all modules in mmdet into the registries
    register_all_modules()

    # TODO: Support inference of image directory.
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta


    if args.mode=='img':
        inference_image(args,model,visualizer)

    elif args.mode=='video':
        inference_video(args,model,visualizer)

    else:
        inference_camera(args,model,visualizer)




if __name__ == '__main__':
    args = parse_args()
    main(args)
