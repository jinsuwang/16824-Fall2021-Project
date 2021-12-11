from pathlib import Path
from typing import Iterable, List, NamedTuple

import cv2
import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Instances
from numpy import ndarray
from torch.utils.data import DataLoader, Dataset


class Prediction(NamedTuple):
    x: float
    y: float
    width: float
    height: float
    score: float
    class_name: str


class ImageDataset(Dataset):

    def __init__(self, imagery: List[Path]):
        self.imagery = imagery

    def __getitem__(self, index) -> ndarray:
        return cv2.imread(self.imagery[index].as_posix())

    def __len__(self):
        return len(self.imagery)


class BatchPredictor:
    def __init__(self, cfg: CfgNode, classes: List[str], batch_size: int, workers: int):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.classes = classes
        self.batch_size = batch_size
        self.workers = workers
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __collate(self, batch):
        data = []
        for image in batch:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            image = self.aug.get_transform(image).apply_image(image)
            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append({"image": image, "height": height, "width": width})
        return data

    def __call__(self, imagery: List[Path]) -> Iterable[List[Prediction]]:
        """[summary]

        :param imagery: [description]
        :type imagery: List[Path]
        :yield: Predictions for each image
        :rtype: [type]
        """
        dataset = ImageDataset(imagery)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.__collate,
            pin_memory=True
        )
        with torch.no_grad():
            for batch in loader:
                results: List[Instances] = self.model(batch)
                yield from [self.__map_predictions(result['instances']) for result in results]

    def __map_predictions(self, instances: Instances):
        instance_predictions = zip(
            instances.get('pred_boxes'),
            instances.get('scores'),
            instances.get('pred_classes')
        )

        predictions = []
        for box, score, class_index in instance_predictions:
            x1 = box[0].item()
            y1 = box[1].item()
            x2 = box[2].item()
            y2 = box[3].item()
            width = x2 - x1
            height = y2 - y1
            prediction = Prediction(
                x1, y1, width, height, score.item(), self.classes[class_index])
            predictions.append(prediction)
        return predictions
