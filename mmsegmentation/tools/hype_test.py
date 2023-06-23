import os
import os.path as osp
import numpy as np
import torch
import cv2

from collections import defaultdict
from tqdm.auto import tqdm

from mmengine.dataset import Compose
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import MODELS


from tools.hype_tools import encode_mask_to_rle, to_csv_file


CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


def _prepare_data(imgs, model, cfg):
    for t in cfg.test_pipeline:
        if t.get("type") in ["CustomLoadAnnotations", "TransposeAnnotations"]:
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]["type"] = "LoadImageFromNDArray"

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data["inputs"].append(data_["inputs"])
        data["data_samples"].append(data_["data_samples"])

    return data, is_batch


def inference(model, data_loader, cfg, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = 29

        for step, test_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            image_name = osp.join(
                test_dict["data_samples"][0].img_path.split("/")[7],
                test_dict["data_samples"][0].img_path.split("/")[8],
            )
            img = cv2.imread(test_dict["data_samples"][0].img_path)

            data, is_batch = _prepare_data(img, model, cfg)

            results = model.test_step(data)

            outputs = results[0].pred_sem_seg.data

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for c, segm in enumerate(outputs):
                rle = encode_mask_to_rle(segm)
                rles.append(rle)
                filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


def test():
    cfg = Config.fromfile(
        "/opt/ml/level2_cv_semanticsegmentation-cv-15/mmsegmentation/configs/hype_custom/swin_large_w12_pre_upernet.py"
    )
    cfg.launcher = "none"
    cfg.work_dir = "/opt/ml/level2_cv_semanticsegmentation-cv-15/mmsegmentation/exp/swin_large_pre_upernet_hand_pretrain"

    runner = Runner.from_cfg(cfg)
    model = MODELS.build(cfg.model)
    test_dataloader = runner.build_dataloader(cfg.test_dataloader)
    model_path = os.path.join(cfg.work_dir, "iter_12000.pth")
    load_checkpoint(model, model_path, map_location="cpu")
    rles, filename_and_class = inference(model, test_dataloader, cfg)

    to_csv_file(rles, filename_and_class)


if __name__ == "__main__":
    test()
