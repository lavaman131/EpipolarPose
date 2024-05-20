from typing import List, Union, Dict, Tuple
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm
from easydict import EasyDict as edict

s_36_flip_pairs = np.array(
    [[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int32
)
s_36_parent_ids = np.array(
    [0, 0, 1, 2, 0, 4, 5, 0, 8, 8, 9, 8, 11, 12, 8, 14, 15], dtype=np.int32
)


def extract_camera_params(
    camera_params: Dict[str, List],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = np.array(camera_params["rotationMatrix"]).T
    T = np.array(camera_params["translationVector"])
    K = np.array(camera_params["IntrinsicMatrix"]).T
    f = np.array([K[0, 0], K[1, 1]])
    c = np.array([K[0, 2], K[1, 2]])
    projection_matrix = np.dot(K, np.hstack((R, T[:, None])))
    return R, T, K, f, c, projection_matrix


def world_to_camera_coords(
    joints_3d: np.ndarray, R: np.ndarray, T: np.ndarray
) -> np.ndarray:
    # Rotate and translate the points to the camera coordinate system with NaNs
    joints_3d = R @ joints_3d.T + T[:, None]
    return joints_3d.T


def camera_to_image_coords(joints_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    # Convert the points to pixel coordinates
    image_coords = K @ joints_3d.T
    image_coords[:2, :] /= image_coords[2, :]
    return image_coords[:2, :].T


def get_bbox_info(
    joints_3d: np.ndarray, R: np.ndarray, T: np.ndarray, K: np.ndarray
) -> Tuple[int, int, int, int]:
    image_coords = camera_to_image_coords(world_to_camera_coords(joints_3d, R, T), K)
    min_x = int(np.nanmin(image_coords[:, 0]))
    max_x = int(np.nanmax(image_coords[:, 0]))
    min_y = int(np.nanmin(image_coords[:, 1]))
    max_y = int(np.nanmax(image_coords[:, 1]))
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width = max_x - min_x
    height = max_y - min_y
    return center_x, center_y, width, height


# Notes: inside each Camera1, ... dirs there are camera_params.json and images_every20
def create_dataset(
    subject_fnames: List[str],  # e.g. ["s1-d1", ...]
    split: str,  # e.g. "train"
    data_dir: Union[
        str, Path
    ],  # e.g. "/projectnb/ivc-ml/Datasets/PoseEstimation/Rat7M"
    save_dir: Union[
        str, Path
    ],  # e.g. "/projectnb/ivc-ml/Datasets/PoseEstimation/Rat7M_processed"
) -> None:
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir

    num_joints = 20
    # Note: contains NaNs
    data = defaultdict(list)
    camera_dir_names = [f"Camera{i}" for i in range(1, 7)]
    for subject_fname in tqdm(subject_fnames):
        subject_id = subject_fname.split("-")[0]
        subject_dir = data_dir.joinpath(subject_fname)
        joints_3d_all = np.load(
            subject_dir.joinpath("mocap_20.npy")
        )  # shape: (num_samples, num_joints * 3)
        joints_3d_all = joints_3d_all.reshape(
            -1, num_joints, 3
        )  # shape: (num_samples, num_joints, 3)
        for camera_idx, camera_dir_name in enumerate(camera_dir_names):
            camera_dir = subject_dir.joinpath(camera_dir_name)
            with open(camera_dir.joinpath("camera_params.json")) as f:
                camera_params = json.load(f)
            R, T, K, f, c, projection_matrix = extract_camera_params(camera_params)
            cam_params = edict(
                {
                    "R": R,
                    "T": T,
                    "K": K,
                    "f": f,
                    "c": c,
                    "projection_matrix": projection_matrix,
                }
            )
            camera_id = camera_idx + 1
            images_dir = camera_dir.joinpath("images")
            for image_path in images_dir.iterdir():
                image_relative_path = image_path.relative_to(data_dir)
                idx_3d = int(image_relative_path.name.split("_")[0])
                joints_3d = joints_3d_all[idx_3d]
                if np.isnan(joints_3d).all():
                    continue
                joints_3d_vis = np.ones_like(joints_3d)
                action = ""
                center_x, center_y, width, height = get_bbox_info(joints_3d, R, T, K)
                sample = {
                    "image": image_relative_path,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height,
                    "joints_3d": joints_3d,
                    "joints_3d_vis": joints_3d_vis,
                    "flip_pairs": s_36_flip_pairs,
                    "parent_ids": s_36_parent_ids,
                    "cam": cam_params,
                    "subject": subject_id,
                    "action": action,
                }
                data[camera_id].append(sample)
    annot_save_dir = save_dir.joinpath("annot")
    annot_save_dir.mkdir(parents=True, exist_ok=True)
    with open(annot_save_dir.joinpath(f"{split}.pkl"), "wb") as f:
        pkl.dump(data, f)


if __name__ == "__main__":
    data_dir = "/projectnb/ivc-ml/Datasets/PoseEstimation/Rat7M"
    save_dir = "/projectnb/ivc-ml/alavaee/data/Rat7M_processed"
    train_subject_fnames = ["s1-d1", "s2-d1", "s2-d2", "s3-d1", "s4-d1"]
    test_subject_fnames = ["s5-d1", "s5-d2"]
    create_dataset(train_subject_fnames, "train-ss", data_dir, save_dir)
    create_dataset(test_subject_fnames, "test", data_dir, save_dir)
