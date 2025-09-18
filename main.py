"""

数据集生成方案

生成新数据集

"""

# std
import json
import math
import random
from pathlib import Path
from typing import Tuple, cast, Any, Optional, Literal
import re
import threading
import multiprocessing
import os

# third_party
import cv2 as cv
import h5py
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from rich import print
from rich.console import Console
import typer
from loguru import logger

"""
COPY FROM: 
    https://github.com/cvlab-epfl/disk/blob/master/disk/data/disk_dataset.py

COMMENTS:
    Use this info to write code.

================================================================================
The datasets are all loaded based on a json file which specifies which tuples
(pairs, triplets, ...) of images are covisible. The structure of the dataset
is as follows:

{
    scene_name_1: {
        image_path: path_to_directory_with_images_for_scene_1,
        depth_path: path_to_directory_with_depths_for_scene_1,
        calib_path: path_to_directory_with_calibs_for_scene_1,
        images: [img_name_1, img_name_2, ...],
        tuples: [[id1_1, id1_2, id1_3], [id2_1, id2_2, id2_3], ...]
    }
}

where 
    * `path_to_directory_with_*_for_scene_*` can be absolute or relative to
      the location of the json file itself.

    * `depth_path` may be missing if you always use DISKDataset with
      no_depth=True

    * `images` lists the file names *with* their extension

    * `tuples` specifies co-visible tuples by their IDs in `images`, that is
      the first tuple above consists of
      [images[id1_1], images[id1_2], images[id1_3]]
      and all tuples are of equal length.
"""


def get_points_from1to2(
        image1, depth1, K1, R1, T1, image2, depth2, K2, R2, T2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    根据两个影像的内参和外参信息，计算得到点的信息并返回

    :param image1:
    :param depth1:
    :param K1:
    :param R1:
    :param T1:
    :param image2:
    :param depth2:
    :param K2:
    :param R2:
    :param T2:
    :return:
    """
    POINTS_RATIO = 20
    POINTS_NUM = int(image1.shape[0] / POINTS_RATIO) * int(
        image1.shape[1] / POINTS_RATIO
    )

    #
    points1 = np.stack(
        [
            np.random.randint(0, image1.shape[1], (POINTS_NUM)),
            np.random.randint(0, image1.shape[0], (POINTS_NUM)),
        ],
        axis=1,
    )

    #
    points1 = points1[depth1[points1[:, 1], points1[:, 0]] > 0, :]

    #

    homo_points1 = np.concatenate(
        [points1, np.ones((points1.shape[0], 1))], axis=1, dtype=np.float32
    )

    space_points = np.linalg.inv(R1) @ (
            np.linalg.inv(K1)
            @ (
                    depth1[
                        homo_points1.astype(np.uint32)[:, 1],
                        homo_points1.astype(np.uint32)[:, 0],
                    ][:, np.newaxis]
                    * homo_points1
            ).T
            - np.stack([T1], 1)
    )
    space_points = space_points.T

    homo_points2 = K2 @ (R2 @ space_points.T + np.stack([T2], 1))
    homo_points2 = homo_points2.T

    homo_points2 = homo_points2 / homo_points2[:, [2]]
    points2 = homo_points2[:, :2]

    # 筛选
    points1 = points1[
        (points2[:, 0] >= 0)
        & (points2[:, 0] < image2.shape[1])
        & (points2[:, 1] >= 0)
        & (points2[:, 1] < image2.shape[0])
        ]
    points2 = points2[
        (points2[:, 0] >= 0)
        & (points2[:, 0] < image2.shape[1])
        & (points2[:, 1] >= 0)
        & (points2[:, 1] < image2.shape[0])
        ]

    # Now Get All the points
    point_depths1 = depth1[points1[:, 1], points1[:, 0]]
    point_depths2 = depth2[
        points2[:, 1].astype(np.uint32), points2[:, 0].astype(np.uint32)
    ]

    return points1, points2, space_points, point_depths1, point_depths2


def get_grid_points(
        image1, depth1, K1, R1, T1, image2, depth2, K2, R2, T2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    根据两个影像的内参和外参信息，计算得到网格化的点的信息并返回

    :param image1:
    :param depth1:
    :param K1:
    :param R1:
    :param T1:
    :param image2:
    :param depth2:
    :param K2:
    :param R2:
    :param T2:
    :return:
    """

    POINTS_RATIO = 8

    #
    points1 = np.stack(
        np.meshgrid(
            np.arange(POINTS_RATIO, image1.shape[1], POINTS_RATIO),
            np.arange(POINTS_RATIO, image1.shape[0], POINTS_RATIO),
        ),
        2,
    ).reshape((-1, 2))

    #
    points1 = points1[depth1[points1[:, 1], points1[:, 0]] > 0, :]

    #
    homo_points1 = np.concatenate(
        [points1, np.ones((points1.shape[0], 1))], axis=1, dtype=np.float32
    )

    space_points = np.linalg.inv(R1) @ (
            np.linalg.inv(K1)
            @ (
                    depth1[
                        homo_points1.astype(np.uint32)[:, 1],
                        homo_points1.astype(np.uint32)[:, 0],
                    ][:, np.newaxis]
                    * homo_points1
            ).T
            - np.stack([T1], 1)
    )
    space_points = space_points.T

    homo_points2 = K2 @ (R2 @ space_points.T + np.stack([T2], 1))
    homo_points2 = homo_points2.T

    homo_points2 = homo_points2 / homo_points2[:, [2]]
    points2 = homo_points2[:, :2]

    # 筛选
    points1 = points1[
        (points2[:, 0] >= 0)
        & (points2[:, 0] < image2.shape[1])
        & (points2[:, 1] >= 0)
        & (points2[:, 1] < image2.shape[0])
        ]
    points2 = points2[
        (points2[:, 0] >= 0)
        & (points2[:, 0] < image2.shape[1])
        & (points2[:, 1] >= 0)
        & (points2[:, 1] < image2.shape[0])
        ]

    # 遮挡筛选
    # 深度图
    point_depths1 = depth1[points1[:, 1], points1[:, 0]]
    point_depths2 = depth2[
        points2[:, 1].astype(np.uint32), points2[:, 0].astype(np.uint32)
    ]

    return points1, points2, space_points, point_depths1, point_depths2


def get_points(
        image1, depth1, K1, R1, T1, image2, depth2, K2, R2, T2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    通过两边同时计算得到对应的空间点

    :param image1:
    :param depth1:
    :param K1:
    :param R1:
    :param T1:
    :param image2:
    :param depth2:
    :param K2:
    :param R2:
    :param T2:
    :return:
    """

    # 各自获取结果
    points1_1, points2_1, space_points_1, point_depths1_1, point_depths2_1 = (
        get_points_from1to2(image1, depth1, K1, R1, T1, image2, depth2, K2, R2, T2)
    )
    points2_2, points1_2, space_points_2, point_depths2_2, point_depths1_2 = (
        get_points_from1to2(image2, depth2, K2, R2, T2, image1, depth1, K1, R1, T1)
    )

    # 汇总结果
    points1 = np.concatenate([points1_1, points1_2], axis=0)
    points2 = np.concatenate([points2_1, points2_2], axis=0)
    space_points = np.concatenate([space_points_1, space_points_2], axis=0)
    point_depths1 = np.concatenate([point_depths1_1, point_depths1_2], axis=0)
    point_depths2 = np.concatenate([point_depths2_1, point_depths2_2], axis=0)

    return points1, points2, space_points, point_depths1, point_depths2


def get_cos(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """
        三个点的位置

    Args:
        A: 空间点
        B: 相机点1
        C: 相机点2

    :param A:
    :param B:
    :param C:
    :return:
    """

    """
    余弦定理的小应用
    """

    a = np.linalg.norm(B - C, 2)
    b = np.linalg.norm(A - C, 2)
    c = np.linalg.norm(A - B, 2)

    ans = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    return ans


def get_pixel_delta(
        point_depth_mean1: float, K1: np.ndarray, point_depth_mean2: float, K2: np.ndarray
) -> float:
    """
    【计算像素差异】

    :param point_depth_mean1:
    :param K1:
    :param point_depth_mean2:
    :param K2:
    :return:
    """
    fx1 = K1[0, 0]
    fy1 = K1[1, 1]
    fx2 = K2[0, 0]
    fy2 = K2[1, 1]

    th = math.pi / 4

    fp1 = (1 / point_depth_mean1) * math.sqrt(
        (fx1 ** 2) * (math.cos(th) ** 2) + (fy1 ** 2) * (math.sin(th) ** 2)
    )
    fp2 = (1 / point_depth_mean2) * math.sqrt(
        (fx2 ** 2) * (math.cos(th) ** 2) + (fy2 ** 2) * (math.sin(th) ** 2)
    )

    fp = math.fabs(fp1 - fp2) / (fp1 + fp2)
    return fp


def get_cache_key2(scene_id: str, img1_name: str, img2_name: str) -> str:
    return f"{scene_id}/{img1_name[:-6]}_{img2_name[:-6]}"


def draw_image_with_points(
        image1: np.ndarray,
        points1: np.ndarray,
        image2: np.ndarray,
        points2: np.ndarray,
        msg: str,
        output_filename: str,
) -> None:
    """
    绘制可视化图像，单纯地进行图像和特征点的展示

    :param image1:
    :param points1:
    :param image2:
    :param points2:
    :param msg:
    :param output_filename:
    :return:
    """

    #
    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    axes[0].imshow(image1[:, :, ::-1])
    axes[0].scatter(points1[:, 0], points1[:, 1], c="r", marker="o", s=1)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(image2[:, :, ::-1])
    axes[1].scatter(points2[:, 0], points2[:, 1], c="r", marker="o", s=1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[0].set_title(f"{image1.shape}")
    axes[1].set_title(f"{image2.shape}")
    fig.suptitle(msg)

    plt.savefig(output_filename)
    plt.close()


def draw_image_little_points(
        image1: np.ndarray,
        points1: np.ndarray,
        image2: np.ndarray,
        points2: np.ndarray,
        output_filename: str,
) -> None:
    """

    :param image1:
    :param points1:
    :param image2:
    :param points2:
    :param output_filename:
    :return:
    """

    #
    ROW_NUM = 6
    COL_NUM = 6

    # draw marker
    image1 = np.copy(image1)
    image2 = np.copy(image2)

    # ？绘制区域
    image1[points1[:, 1].astype(np.uint32), points1[:, 0].astype(np.uint32)] = (
        0,
        0,
        255,
    )
    image2[points2[:, 1].astype(np.uint32), points2[:, 0].astype(np.uint32)] = (
        0,
        0,
        255,
    )

    fig, axes = plt.subplots(ROW_NUM, COL_NUM)
    fig = cast(Figure, fig)
    fig.set_figwidth(20)
    fig.set_figheight(20)

    for row_idx in range(ROW_NUM):
        for col_idx in range(0, COL_NUM, 2):

            # 绘制 subfigure

            while True:
                idx = random.randint(0, points1.shape[0] - 1)

                t1 = int(points1[idx, 1] - 16)
                b1 = int(points1[idx, 1] + 16 + 1)
                l1 = int(points1[idx, 0] - 16)
                r1 = int(points1[idx, 0] + 16 + 1)

                if not (
                        (0 <= t1 < image1.shape[0])
                        and (0 <= b1 < image1.shape[0])
                        and (0 <= l1 < image1.shape[1])
                        and (0 <= r1 < image1.shape[1])
                ):
                    continue

                t2 = int(points2[idx, 1] - 16)
                b2 = int(points2[idx, 1] + 16)
                l2 = int(points2[idx, 0] - 16)
                r2 = int(points2[idx, 0] + 16)
                if not (
                        (0 <= t2 < image1.shape[0])
                        and (0 <= b2 < image1.shape[0])
                        and (0 <= l2 < image1.shape[1])
                        and (0 <= r2 < image1.shape[1])
                ):
                    continue

                # draw
                sub_img1 = np.copy(image1[t1:b1, l1:r1, ::-1])
                sub_img1[16, 16, 1] = 255
                axes[row_idx, col_idx].imshow(sub_img1)
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])

                sub_img2 = np.copy(image2[t2:b2, l2:r2, ::-1])
                sub_img2[16, 16, 1] = 255
                axes[row_idx, col_idx + 1].imshow(sub_img2)
                axes[row_idx, col_idx + 1].set_xticks([])
                axes[row_idx, col_idx + 1].set_yticks([])
                break

    plt.savefig(output_filename)
    plt.close()


class MegadepthCacheManager:
    def __init__(self):
        """
        Megadepth 数据集处理的缓存机制

        """
        self.cache_dir = Path(".cache")
        logger.info(f"cache path = {self.cache_dir.absolute()}")
        if not self.cache_dir.exists():
            logger.info("cache is not exists, create it.")
            self.cache_dir.mkdir(parents=True)

    def set(self, cache_key: str, data: dict[str, Any]) -> None:
        cache_filename = self.cache_dir / (cache_key + ".json")
        cache_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_filename, "w") as f:
            json.dump({"data": data}, f)

    def get(self, cache_key: str) -> Optional[dict[str, Any]]:
        cache_filename = self.cache_dir / (cache_key + ".json")
        if not cache_filename.exists():
            return None
        try:
            with open(cache_filename, "r") as f:
                return json.load(f)["data"]
        except Exception as e:
            logger.error(f"cache file {cache_filename} is broken, delete it.")
            os.remove(cache_filename)


class MegadepthRecoder:
    def __init__(self):
        # ?
        self.data = {
            "ADI": {"low": 0, "medium": 0, "high": 0, "ultra": 0},
            "PDI": {"low": 0, "medium": 0, "high": 0, "ultra": 0},
            "SDI": {"low": 0, "medium": 0, "high": 0, "ultra": 0},
        }

        # ?
        self.data2 = {
            "ADI": {"Low": [], "Medium": [], "High": [], "Ultra": []},
            "PDI": {"Low": [], "Medium": [], "High": [], "Ultra": []},
            "SDI": {"Low": [], "Medium": [], "High": [], "Ultra": []},
        }

    def record2(self, data: dict[str, Any]) -> None:
        def sub_record(DI_type: Literal["ADI", "PDI", "SDI"], DI: float) -> None:
            if DI < 0.2:
                self.data[DI_type]["low"] += 1
                self.data2[DI_type]["Low"].append(data)
            elif DI < 0.4:
                self.data[DI_type]["medium"] += 1
                self.data2[DI_type]["Medium"].append(data)
            elif DI < 0.6:
                self.data[DI_type]["high"] += 1
                self.data2[DI_type]["High"].append(data)
            else:
                self.data[DI_type]["ultra"] += 1
                self.data2[DI_type]["Ultra"].append(data)

        sub_record("ADI", data["ADI"])
        sub_record("PDI", data["PDI"])
        sub_record("SDI", data["SDI"])

    def display(self) -> None:
        print(self.data)

    def save_data(self, data_path: Optional[Path] = None) -> None:
        if data_path is None:
            data_path = Path("./megadepth_recorder.json")

        with open(data_path, "w") as f:
            json.dump(self.data2, f)


class NewDatasetGenerator:
    def __init__(self, megadepth_path: Path):
        """

        :param megadepth_path:
        """

        # 数据集路径
        self.dataset_path = megadepth_path
        self.dataset_json_path = self.dataset_path / "dataset.json"

        # MegaDepth数据集的核心JSON文件信息读取
        with open(self.dataset_json_path, "r") as f:
            data = json.load(f)
            self.data = data

        self.cache_manager = MegadepthCacheManager()
        self.recoder = MegadepthRecoder()

    def run_record(self):
        """
        对缓存中的JSON文件数值进行计算，并根据数值方法进行分配汇总

        """

        scene_ids = list(self.data.keys())

        for scene_id in scene_ids:

            # 缓存路径
            filepaths = list(Path(f".cache/{scene_id}").glob("*"))

            # 对于每个JSON文件，打开并进行根据DI进行记录
            for filepath in tqdm.tqdm(filepaths):
                with open(filepath, "r") as f:
                    data = json.load(f)
                    self.recoder.record2(data["data"])

            # 展示信息并保存
            self.recoder.display()
            self.recoder.save_data(Path(f"{scene_id}.json"))

    def run_dataset(self):
        """
        用于生成需要测试的图像对的对应数据

        Returns:
        """

        scene_ids = list(self.data.keys())
        for scene_idx, scene_id in enumerate(scene_ids):
            logger.info(f"Processing {scene_idx}/{len(scene_ids)}: Scene ID = {scene_id}")

            scene = self.data[scene_id]
            tuples = scene["tuples"]

            logger.info(f"[SceneID: {scene_id}, {scene_idx}/{len(scene_ids)}]")

            # 处理每个tuple
            for tup_idx, tup in tqdm.tqdm(enumerate(tuples), total=len(tuples)):
                logger.info(f"[SceneID: {scene_id}, {scene_idx}/{len(scene_ids)}] [Tup {tup_idx}/{len(tuples)}]")
                try:
                    self._process_tup(scene_id, tup_idx)
                except Exception as e:
                    logger.error(f"[SceneID: {scene_id}, {scene_idx}/{len(scene_ids)}] [Tup {tup_idx}/{len(tuples)}] Error: {e}")
                    continue
                else:
                    logger.info(f"[SceneID: {scene_id}, {scene_idx}/{len(scene_ids)}] [Tup {tup_idx}/{len(tuples)}] Done!")
            break

    def run_dataset_mp(self):
        """
        用于生成需要测试的图像对的对应数据

        Returns:
        """
        pl = multiprocessing.Pool(12)

        scene_ids = list(self.data.keys())
        for scene_idx, scene_id in enumerate(scene_ids):
            logger.info(f"Processing {scene_idx}/{len(scene_ids)}: Scene ID = {scene_id}")
            pl.apply_async(self._process_scene, args=(scene_id,))

        pl.close()
        pl.join()

    def run_dataset_mp2(self):
        """
        用于生成需要测试的图像对的对应数据

        Returns:
        """
        scene_ids = list(self.data.keys())
        for scene_idx, scene_id in enumerate(scene_ids):
            logger.info(f"Processing {scene_idx}/{len(scene_ids)}: Scene ID = {scene_id}")
            self._process_scene_mp2(scene_id)
            # break

    def compute_delta(
            self, image1, depth1, K1, R1, T1, image2, depth2, K2, R2, T2
    ) -> tuple[float, float, float]:
        """计算两幅影像之间的差异"""

        # points1, points2, space_points, point_depths1, point_depths2 = get_points(image1, depth1, K1,
        #                                                                           R1,
        #                                                                           T1, image2, depth2,
        #                                                                           K2,
        #                                                                           R2, T2)

        points1, points2, space_points, point_depths1, point_depths2 = get_grid_points(
            image1, depth1, K1, R1, T1, image2, depth2, K2, R2, T2
        )

        point_mean = space_points.mean(axis=0)

        angle_delta = get_cos(point_mean, T1, T2)

        #
        pixel_delta = get_pixel_delta(
            float(point_depths1.mean()), K1, float(point_depths2.mean()), K2
        )

        w1 = 1.0
        w2 = 1.0

        # the Index
        ADI = angle_delta
        PDI = pixel_delta
        SDI = w1 * ADI + w2 * PDI

        return ADI, PDI, SDI

    def _get_tup_infos(
            self, scene_id: str, tuple_id: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        images = []
        depths = []
        Ks = []
        Rs = []
        Ts = []

        scene = self.data[scene_id]
        image_dir_path = scene["image_path"]
        depth_dir_path = scene["depth_path"]
        calib_dir_path = scene["calib_path"]
        image_names = scene["images"]
        tuples = scene["tuples"]

        # 由于定义问题，这里tuple的length只能是3
        for img_id in tuples[tuple_id]:
            # 图像信息读取
            image_path = (self.dataset_path / image_dir_path / image_names[img_id])
            images.append(cv.imread(str(image_path)))

            # 深度图信息读取
            depth_path = (self.dataset_path / depth_dir_path / f"{image_names[img_id][:-4]}.h5")
            with h5py.File(depth_path, "r") as f:
                depths.append(np.array(f["depth"]))

            # 相机信息读取   K R T
            calib_path = (self.dataset_path / calib_dir_path / f"calibration_{image_names[img_id]}.h5")
            with h5py.File(calib_path, "r") as f:
                Ks.append(np.array(f["K"]))
                Rs.append(np.array(f["R"]))
                Ts.append(np.array(f["T"]))

        return images, depths, Ks, Rs, Ts


    def _process_scene(self, scene_id: str):
        scene = self.data[scene_id]
        tuples = scene["tuples"]

        for tup_idx, tup in enumerate(tuples):
            self._process_tup_exp(scene_id, tup_idx)

    def _process_scene_mp2(self, scene_id: str):
        scene = self.data[scene_id]
        tuples = scene["tuples"]

        ct = multiprocessing.cpu_count()
        ps = []
        for cti in range(ct):
            p = multiprocessing.Process(target=self._process_scene_mp2_helper, args=(scene_id, cti, len(tuples), ct))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    def _process_scene_mp2_helper(self, scene_id: str, tup_idx_start:int, tup_idx_end:int, tup_idx_step:int):
        for ti in range(tup_idx_start, tup_idx_end, tup_idx_step):
            self._process_tup_exp(scene_id, ti)

    def _process_tup_exp(self, scene_id: str, tuple_id: int):
        try:
            logger.info(f"[SceneID: {scene_id}] [Tup {tuple_id}]")
            self._process_tup(scene_id, tuple_id)
        except Exception as e:
            logger.error(
                f"[SceneID: {scene_id}] [Tup {tuple_id}] Error: {e}")

    def _process_tup(self, scene_id: str, tuple_id: int):
        """
        处理这两个idx的图像的信息
        """
        images, depths, Ks, Rs, Ts = self._get_tup_infos(scene_id, tuple_id)

        self._process_tup_12(scene_id, tuple_id, 0, 1, images, depths, Ks, Rs, Ts)
        self._process_tup_12(scene_id, tuple_id, 0, 2, images, depths, Ks, Rs, Ts)
        self._process_tup_12(scene_id, tuple_id, 1, 2, images, depths, Ks, Rs, Ts)

    def _process_tup_12(self, scene_id: str, tuple_id: int, idx1: int, idx2: int, images, depths, Ks, Rs, Ts):

        # 缓存key
        scene = self.data[scene_id]
        image_dir_path = scene["image_path"]
        depth_dir_path = scene["depth_path"]
        calib_dir_path = scene["calib_path"]
        image_names = scene["images"]
        tuples = scene["tuples"][tuple_id]

        logger.info(f"processing {image_names[tuples[idx1]]} and {image_names[tuples[idx2]]}")

        cache_key = get_cache_key2(scene_id, image_names[tuples[idx1]], image_names[tuples[idx2]])
        logger.info(f"cache_key: {cache_key}")

        DIs = self.cache_manager.get(cache_key)
        if DIs is not None:
            logger.info(f"cache_key: {cache_key}, targeted")
            return

        logger.info(f"cache_key: {cache_key}, not targeted")
        ADI, PDI, SDI = self.compute_delta(
            images[idx1],
            depths[idx1],
            Ks[idx1],
            Rs[idx1],
            Ts[idx1],
            images[idx2],
            depths[idx2],
            Ks[idx2],
            Rs[idx2],
            Ts[idx2],
        )

        logger.info(f"ADI={ADI}, PDI={PDI}, SDI={SDI}")

        self.cache_manager.set(
            cache_key,
            {
                "img1_path": str(image_dir_path + "/" + image_names[tuples[idx1]]),
                "img1_depth_path": str(depth_dir_path + "/" + f"{image_names[tuples[idx1]][:-4]}.h5"),
                "img1_calib_path": str(calib_dir_path + "/" + f"calibration_{image_names[tuples[idx1]]}.h5"),
                "img2_path": str(image_dir_path + "/" + image_names[tuples[idx2]]),
                "img2_depth_path": str(depth_dir_path + "/" + f"{image_names[tuples[idx2]][:-4]}.h5"),
                "img2_calib_path": str(calib_dir_path + "/" + f"calibration_{image_names[tuples[idx2]]}.h5"),
                "ADI": ADI,
                "PDI": PDI,
                "SDI": SDI,
            },
        )


################################################################################

app = typer.Typer(no_args_is_help=True)


@app.command()
def gen_msd(
        megadepth_path: Optional[Path] = typer.Argument(exists=True, dir_okay=True, readable=True,
                                                        help="The path of MegaDepth")
) -> None:
    logger.info("MegaDepth path = {}".format(megadepth_path))

    md = NewDatasetGenerator(megadepth_path=megadepth_path)

    # md.run_dataset()
    # md.run_dataset_mp()
    # md.run_dataset_mp2()
    md.run_record()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
