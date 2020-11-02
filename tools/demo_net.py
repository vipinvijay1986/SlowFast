#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import time
import torch
import tqdm
from slowfast.utils.misc import get_class_names
from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer
import pandas as pd
from PIL import Image
logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    # Start reading frames.
    frame_provider.start()
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        model.put(task)
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue

    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    class_names, _, _ = get_class_names(cfg.DEMO.LABEL_FILE_PATH, None, None)
    print(class_names)
    columns=['Task_id', 'Frame_file', 'Bbox_x0', 'Bbox_x1', 'Bbox_y0', 'Bbox_y1', 'Action', 'Scores']
    columns.extend(class_names)
    df_frames = pd.DataFrame(columns=columns)
    
    # AVA format-specific visualization with precomputed boxes.
    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
    else:
        start = time.time()
        if cfg.DEMO.THREAD_ENABLE:
            frame_provider = ThreadVideoManager(cfg)
        else:
            frame_provider = VideoManager(cfg)

        for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
            if cfg.DEMO.SAVE_ACTION_FILE_PATH !='':
                    
                print(task.id ,"-->",task.frames.shape,task.action_preds.shape,task.bboxes.shape,task.clip_vis_size,task.num_buffer_frames)
                middle_frame = task.frames[len(task.frames) // 2]
                bbox =task.bboxes
                print("middle_frame.shape",middle_frame.shape)
                print("bbox.shape",bbox.shape)
                preds =task.action_preds

                top_scores, top_classes, labels = [], [],[]
                for pred in preds :
                    mask = pred >= 0.7
                    print(pred.shape)
                    top_scores.append(pred[mask].tolist())
                    top_class = torch.squeeze(torch.nonzero(mask), dim=-1).tolist()
                    top_classes.append(top_class)
                    lbls= [class_names[i] for i in top_class]
                    labels.append(lbls)
                print("top_scores",top_scores)
                #print("top_classes",top_classes)
                print("labels",labels)
                #print("preds.shape",preds.shape)
                print('_________________________')
                for b in bbox :
                    x0, y0, x1, y1 = b
                    x0 = int(x0.item())
                    x1 = int(x1.item())
                    y0 = int(y0.item())
                    y1 = int(y1.item())
                    print('\t', x0, x1, y0, y1,)
                print('************************')
                bbox =task.resized_boxes
                print('_________RESIZED BOX________________')
                for b in bbox :
                    x0, y0, x1, y1 = b
                    x0 = int(x0.item())
                    x1 = int(x1.item())
                    y0 = int(y0.item())
                    y1 = int(y1.item())
                    print('\t', x0, x1, y0, y1,)
                print('************************')
                frame_file = cfg.DEMO.SAVE_ACTION_FILE_PATH+"/frame_withbounds_" + str(task.id) + '.jpg'
                im = Image.fromarray(task.frames[task.key_frame_index])
                im.save(frame_file)
                key_frame_file = cfg.DEMO.SAVE_ACTION_FILE_PATH+"/frame_" + str(task.id) + '.jpg'
                im = Image.fromarray(task.key_frame)
                im.save(key_frame_file)
                for idx, box in enumerate(task.resized_boxes) :
                    x0, y0, x1, y1 = box
                    x0 = int(x0.item())
                    x1 = int(x1.item())
                    y0 = int(y0.item())
                    y1 = int(y1.item())
                    print('No of rows={0}, INSERTING NEW ROW...'.format(df_frames.index))
                    row_data =[task.id, frame_file, x0, x1, y0, y1, labels[idx], top_scores[idx]]
                    row_data.extend(preds[idx].tolist())
                    df_frames.loc[len(df_frames.index)] = row_data
            frame_provider.display(task)

        frame_provider.join()
        frame_provider.clean()
        if cfg.DEMO.SAVE_ACTION_FILE_PATH !='':
            print(df_frames)
            frame_csv = cfg.DEMO.SAVE_ACTION_FILE_PATH+'/frames.csv'
            df_frames.to_csv(frame_csv)
        logger.info("Finish demo in: {}".format(time.time() - start))

