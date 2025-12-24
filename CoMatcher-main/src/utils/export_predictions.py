"""
Export the predictions of a model for a given dataloader (e.g. ImageFolder).
Use a standalone script with `python3 -m dsfm.scipts.export_predictions dir`
or call from another script.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .tensor import batch_to_device


@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("lines"):
                idx = k.replace("lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("orig_lines"):
                idx = k.replace("orig_lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}  # b=1

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        try:
            name = data["name"][0]
            grp = hfile.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
        except RuntimeError:
            continue

        del pred
    hfile.close()
    return output_file






@torch.no_grad()
def export_multiview_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.endswith("keypoints"):
                if k.startswith("target"):
                    scales = 1.0 / (data["target_view"]["scales"])
                    pred[k] = pred[k] * scales[None]
                else:
                    scales = 1.0 / (data["source_views"]["scales"])
                    pred[k] = pred[k] * scales[None].transpose(1, 2)

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}  # b=1
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        for idx in range(2, 7):
            loader_idx = data["idx"].item()
            name = data["scene"][0] + f'/{5 * loader_idx+ idx - 2}.ppm'
            pred_dict = {
                "keypoints0": pred["target_keypoints"],
                "keypoints1": pred["source_keypoints"][idx - 2],
                "keypoint_scores0": pred["target_keypoint_scores"],
                "keypoint_scores1": pred["source_keypoint_scores"][idx - 2],
                "matches0": pred["t2s_matches"][idx - 2],
                "matches1": pred["s2t_matches"][idx - 2],
                "matching_scores0": pred["t2s_matches_scores"][idx - 2],
                "matching_scores1": pred["s2t_matches_scores"][idx - 2]
            }
            try:
                grp = hfile.create_group(name)
                for k, v in pred_dict.items():
                    grp.create_dataset(k, data=v)
            except RuntimeError:
                continue
        del pred
    hfile.close()
    return output_file




@torch.no_grad()
def export_multiview_predictions_mega(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.endswith("keypoints"):
                if k.startswith("target"):
                    scales = 1.0 / (data["target_view"]["scales"])
                    pred[k] = pred[k] * scales[None]
                else:
                    scales = 1.0 / (data["source_views"]["scales"])
                    pred[k] = pred[k] * scales[None].transpose(1, 2)

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}  # b=1
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        names = data["names"][0].split("|")
        for i in range(3):
            name = names[i]
            pred_dict = {
                "keypoints0": pred["target_keypoints"],
                "keypoints1": pred["source_keypoints"][i],
                "keypoint_scores0": pred["target_keypoint_scores"],
                "keypoint_scores1": pred["source_keypoint_scores"][i],
                "matches0": pred["t2s_matches"][i],
                "matches1": pred["s2t_matches"][i],
                "matching_scores0": pred["t2s_matches_scores"][i],
                "matching_scores1": pred["s2t_matches_scores"][i]
            }
            try:
                grp = hfile.create_group(name)
                for k, v in pred_dict.items():
                    grp.create_dataset(k, data=v)
            except RuntimeError:
                continue
        del pred
    hfile.close()
    return output_file