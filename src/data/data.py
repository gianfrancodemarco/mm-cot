import json
import os

import numpy as np
from zipfile import ZipFile
from src import constants

def load_data_std(args):
    problems = json.load(
        open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    pid_splits = json.load(
        open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    qids = get_qids(args, captions, pid_splits, problems)
    return problems, qids,


def load_data_img(args):
    problems = json.load(
        open(constants.SCIENCEQA_PROBLEMS_PATH))
    pid_splits = json.load(
        open(constants.SCIENCEQA_PID_SPLITS))
    captions = json.load(open(args.caption_file))["captions"]
    name_maps = json.load(open(constants.SCIENCEQA_NAME_MAP))

    # check
    if args.img_type == "resnet":
        image_features = np.load(constants.SCIENCEQA_RESNET)
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load(constants.SCIENCEQA_CLIP)
    elif args.img_type == "detr":
        image_features = np.load(constants.SCIENCEQA_DETR)
    else:
        image_features = np.load(constants.SCIENCEQA_DETR)
    print("img_features size: ", image_features.shape)

    qids = get_qids(args, captions, pid_splits, problems)
    return problems, qids, name_maps, image_features


def get_qids(args, captions, pid_splits, problems):
    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""
    train_qids = pid_splits['%s' % args.train_split]
    val_qids = pid_splits['%s' % args.val_split]
    test_qids = pid_splits['%s' % args.test_split]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")
    qids = {'train': train_qids, 'val': val_qids, 'test': test_qids}

    return qids


def unzip_folder(folder_name: str = None, destination_path: str = "./"):
    if not folder_name:
        raise NotADirectoryError

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    with ZipFile(f"{folder_name}.zip", 'r') as zip_obj:
        zip_obj.extractall(path=destination_path)
