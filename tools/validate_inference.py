# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Qualify inference against a CPU reference using the bundled image/video assets.

Run from the repository root: python -m tools.validate_inference --device mps.
This measures a small regression workload, not dataset accuracy or peak memory.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment


def mask_iou(first, second):
    """Return binary-mask intersection over union, requiring identical shapes."""
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Masks must have identical spatial shapes")
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 1.0


def matched_mask_fraction(reference, candidate, threshold):
    """Find the maximum one-to-one matching above the required IoU threshold."""
    if not reference:
        raise ValueError("The reference must produce automatic masks")
    if not candidate:
        return 0.0
    agreement = np.array(
        [[mask_iou(first, second) for second in candidate] for first in reference]
    )
    acceptable = agreement >= threshold
    rows, columns = linear_sum_assignment(acceptable, maximize=True)
    return float(acceptable[rows, columns].sum() / len(reference))


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    start = time.perf_counter()
    value = function()
    synchronize(device)
    return value, time.perf_counter() - start


def cleaned_memory(device):
    gc.collect()
    synchronize(device)
    if device.type == "mps":
        return {
            "live_bytes": torch.mps.current_allocated_memory(),
            "driver_bytes": torch.mps.driver_allocated_memory(),
        }
    if device.type == "cuda":
        return {
            "live_bytes": torch.cuda.memory_allocated(device),
            "driver_bytes": torch.cuda.memory_reserved(device),
        }
    return {}


def require_mask(mask, shape):
    if mask.shape != shape or not np.isfinite(mask).all() or not np.any(mask):
        raise RuntimeError(f"Expected a finite, nonempty mask with shape {shape}")


def qualify_image(args, device, image):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.utils.device import inference_context

    model = build_sam2(args.config, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model, max_hole_area=8, max_sprinkle_area=8)

    def predict():
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[500, 375]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=False,
        )
        if not np.isfinite(scores).all() or not np.isfinite(logits).all():
            raise RuntimeError("Image prediction produced nonfinite scores or logits")
        mask = masks[0] > 0
        require_mask(mask, image.shape[:2])
        return mask

    with torch.inference_mode(), inference_context(device):
        _, embedding_seconds = timed(device, lambda: predictor.set_image(image))
        _, warmup_seconds = timed(device, predict)
        samples = [timed(device, predict) for _ in range(args.sessions)]
        mask = samples[-1][0]
        predictor.reset_predictor()
        generator = SAM2AutomaticMaskGenerator(
            model, points_per_side=8, points_per_batch=4, min_mask_region_area=0
        )
        annotations, automatic_seconds = timed(
            device, lambda: generator.generate(image)
        )
        automatic = [annotation["segmentation"] for annotation in annotations]
        if not automatic:
            raise RuntimeError("Automatic mask generation produced no masks")
        for annotation in annotations:
            require_mask(annotation["segmentation"], image.shape[:2])
            if not np.isfinite(
                [annotation["predicted_iou"], annotation["stability_score"]]
            ).all():
                raise RuntimeError("Automatic masks contain nonfinite scores")
    evidence = {
        "embedding_seconds": embedding_seconds,
        "prompt_warmup_seconds": warmup_seconds,
        "cached_prompt_seconds": [seconds for _, seconds in samples],
        "automatic_seconds": automatic_seconds,
        "automatic_mask_count": len(automatic),
        "model_input_size": model.image_size,
    }
    return mask, automatic, evidence


def video_session(
    predictor, video_path, device, reuse=False, offload_state=False, offload_frames=None
):
    options = {"offload_state_to_cpu": offload_state}
    if offload_frames is not None:
        options["offload_video_to_cpu"] = offload_frames
    state = predictor.init_state(str(video_path), **options)
    expected_offload = (
        device.type == "mps" if offload_frames is None else offload_frames
    )
    if state["offload_video_to_cpu"] != expected_offload:
        raise RuntimeError("Video frame storage did not honor the selected policy")
    shape = (state["video_height"], state["video_width"])

    def track():
        for object_id, point in [(2, [200, 300]), (3, [400, 150])]:
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=object_id,
                points=np.array([point], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
        masks = {}
        synchronize(device)
        start = time.perf_counter()
        for frame, object_ids, logits in predictor.propagate_in_video(state):
            if list(object_ids) != [2, 3] or not torch.isfinite(logits).all().item():
                raise RuntimeError(
                    "Video prediction lost object IDs or produced nonfinite logits"
                )
            for index, object_id in enumerate(object_ids):
                mask = (logits[index, 0] > 0).cpu().numpy()
                require_mask(mask, shape)
                masks[(frame, object_id)] = mask
        synchronize(device)
        seconds = time.perf_counter() - start
        if len(masks) != state["num_frames"] * 2:
            raise RuntimeError("Video prediction did not return every object/frame")
        if device.type == "mps":
            for outputs in state["output_dict_per_obj"].values():
                for frames in outputs.values():
                    for output in frames.values():
                        memory = output["maskmem_features"]
                        if memory is not None and memory.dtype != torch.float32:
                            raise RuntimeError("MPS video memory must remain float32")
        return masks, seconds

    masks, seconds = track()
    predictor.reset_state(state)
    if state["obj_ids"] or state["output_dict_per_obj"]:
        raise RuntimeError("Reset did not clear video objects and tracking results")
    if reuse:
        repeated, _ = track()
        if any(mask_iou(masks[key], repeated[key]) < 0.98 for key in masks):
            raise RuntimeError("Reset/reuse changed the video masks")
        predictor.reset_state(state)
    return masks, seconds


def qualify_video(args, device, video_path):
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.utils.device import inference_context

    predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)
    with torch.inference_mode(), inference_context(device):
        _, warmup_seconds = timed(
            device, lambda: video_session(predictor, video_path, device, reuse=True)
        )
        cleaned_memory(device)
        samples, allocations, session_masks = [], [], []
        for session in range(args.sessions):
            offload_frames = [None, False, True][session % 3]
            (masks, propagation_seconds), total_seconds = timed(
                device,
                lambda: video_session(
                    predictor,
                    video_path,
                    device,
                    offload_state=session % 2 == 1,
                    offload_frames=offload_frames,
                ),
            )
            session_masks.append(masks)
            samples.append(
                {
                    "total_seconds": total_seconds,
                    "propagation_seconds": propagation_seconds,
                    "frames_per_second": args.frames / propagation_seconds,
                    "offload_state_to_cpu": session % 2 == 1,
                    "offload_video_to_cpu": offload_frames,
                }
            )
            allocations.append(cleaned_memory(device))
    return session_masks, {
        "warmup_and_reset_reuse_seconds": warmup_seconds,
        "sessions": samples,
        "cleaned_session_memory": allocations,
    }


def environment(checkpoint):
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    details = {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "checkpoint_sha256": digest.hexdigest(),
    }
    if platform.system() == "Darwin":
        for name, key in [
            ("machdep.cpu.brand_string", "chip"),
            ("hw.memsize", "ram_bytes"),
        ]:
            result = subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                details[key] = result.stdout.strip()
    return details


def validate(args):
    from sam2.utils.device import get_device

    device, reference = get_device(args.device), get_device(args.reference_device)
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Unset PYTORCH_ENABLE_MPS_FALLBACK for MPS qualification")
    image = np.array(Image.open(args.image).convert("RGB"))
    frames = sorted(
        (
            path
            for path in args.video_dir.iterdir()
            if path.suffix.lower() in (".jpg", ".jpeg")
        ),
        key=lambda path: int(path.stem),
    )[: args.frames]
    if len(frames) != args.frames:
        raise ValueError(f"Need at least {args.frames} JPEG frames")
    result = {
        "environment": environment(args.checkpoint),
        "device": str(device),
        "reference_device": str(reference),
        "settings": {
            "config": args.config,
            "checkpoint": str(args.checkpoint),
            "image": str(args.image),
            "image_shape": list(image.shape),
            "image_prompt": [[500, 375]],
            "image_cleanup_area": 8,
            "automatic_points_per_side": 8,
            "automatic_points_per_batch": 4,
            "automatic_min_mask_region_area": 0,
            "video_frames": [path.name for path in frames],
            "video_prompts": {"2": [[200, 300]], "3": [[400, 150]]},
            "video_frame_storage": "default, explicit device, explicit CPU",
        },
        "limitations": [
            "JPEG video only; MP4 decoder not qualified",
            "No peak-memory measurement",
            "Small regression workload; not dataset accuracy",
        ],
        "backends": {},
    }
    with TemporaryDirectory(prefix="sam2-qualification-") as directory:
        video_path = Path(directory)
        for path in frames:
            shutil.copyfile(path, video_path / path.name)
        outputs = {}
        for backend in dict.fromkeys((reference, device)):
            print(f"Qualifying {backend}: image and automatic masks", flush=True)
            mask, automatic, image_evidence = qualify_image(args, backend, image)
            cleaned_memory(backend)
            print(f"Qualifying {backend}: video and repeated sessions", flush=True)
            video, video_evidence = qualify_video(args, backend, video_path)
            result["backends"][str(backend)] = {
                "image": image_evidence,
                "video": video_evidence,
            }
            outputs[str(backend)] = (mask, automatic, video)
            cleaned_memory(backend)
    first, second = outputs[str(reference)], outputs[str(device)]
    image_agreement = mask_iou(first[0], second[0])
    automatic_agreement = matched_mask_fraction(first[1], second[1], 0.95)
    baseline = first[2][0]
    session_agreement = {}
    for backend, (_, _, sessions) in outputs.items():
        session_agreement[backend] = []
        for masks in sessions:
            if masks.keys() != baseline.keys():
                raise RuntimeError(
                    "Video sessions returned different object/frame sets"
                )
            session_agreement[backend].append(
                {
                    f"{frame}:{object_id}": mask_iou(mask, masks[(frame, object_id)])
                    for (frame, object_id), mask in baseline.items()
                }
            )
    video_agreement = {
        key: min(session[key] for session in session_agreement[str(device)])
        for key in session_agreement[str(device)][0]
    }
    memory = result["backends"][str(device)]["video"]["cleaned_session_memory"]
    growth = max(
        (
            sample.get("live_bytes", 0) - memory[0].get("live_bytes", 0)
            for sample in memory
        ),
        default=0,
    )
    result["agreement"] = {
        "image_iou": image_agreement,
        "automatic_matched_fraction": automatic_agreement,
        "video_iou": video_agreement,
        "video_sessions": session_agreement,
    }
    result["gates"] = {
        "image_iou_at_least_0.99": image_agreement >= 0.99,
        "automatic_fraction_at_least_0.90_at_iou_0.95": automatic_agreement >= 0.9,
        "video_iou_at_least_0.98": all(
            min(session.values()) >= 0.98
            for sessions in session_agreement.values()
            for session in sessions
        ),
        "three_repeated_sessions": args.sessions >= 3,
        "live_memory_growth_at_most_16_mib": growth <= 16 * 1024**2,
    }
    result["live_memory_growth_bytes"] = growth
    result["passed"] = all(result["gates"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/sam2.1_hiera_tiny.pt")
    )
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument(
        "--image", type=Path, default=Path("notebooks/images/truck.jpg")
    )
    parser.add_argument(
        "--video-dir", type=Path, default=Path("notebooks/videos/bedroom")
    )
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 2 or args.sessions < 1:
        parser.error("--frames must be >= 2 and --sessions must be >= 1")
    result = validate(args)
    serialized = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
