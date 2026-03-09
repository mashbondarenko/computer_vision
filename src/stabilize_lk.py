#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return curve.copy()
    pad = np.pad(curve, (radius, radius), mode="reflect")
    kernel = np.ones(2 * radius + 1, dtype=np.float32) / (2 * radius + 1)
    return np.convolve(pad, kernel, mode="same")[radius:-radius]


def smooth_trajectory(traj: np.ndarray, radius: int) -> np.ndarray:
    out = traj.copy()
    for i in range(traj.shape[1]):
        out[:, i] = moving_average(traj[:, i], radius)
    return out


def normalize_to_rigid(m: np.ndarray | None) -> np.ndarray:
    if m is None:
        return np.eye(3, dtype=np.float32)
    a, tx = float(m[0, 0]), float(m[0, 2])
    c, ty = float(m[1, 0]), float(m[1, 2])
    s = math.sqrt(a * a + c * c)
    if s < 1e-6:
        cos_t, sin_t = 1.0, 0.0
    else:
        cos_t, sin_t = a / s, c / s
    return np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty], [0.0, 0.0, 1.0]], dtype=np.float32)


def mat_from_params(tx: float, ty: float, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]], dtype=np.float32)


def params_from_mat(m: np.ndarray) -> np.ndarray:
    return np.array([m[0, 2], m[1, 2], math.atan2(m[1, 0], m[0, 0])], dtype=np.float32)


def estimate_motion(video_path: str, scale: float, max_corners: int, quality: float, min_dist: int,
                    lk_win: tuple[int, int], lk_levels: int, ransac_thresh: float):
    cv2.setNumThreads(0)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    h, w = prev.shape[:2]
    prev_small = cv2.resize(prev, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    pairwise = []
    tracked_counts = []
    inlier_ratios = []

    while True:
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=max_corners,
            qualityLevel=quality,
            minDistance=min_dist,
            blockSize=3,
            useHarrisDetector=False,
        )

        ret, curr = cap.read()
        if not ret:
            break

        curr_small = cv2.resize(curr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        rigid = np.eye(3, dtype=np.float32)
        tracked = 0
        inliers = 0

        if prev_pts is not None and len(prev_pts) >= 8:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                prev_pts,
                None,
                winSize=lk_win,
                maxLevel=lk_levels,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if curr_pts is not None and status is not None:
                mask = status.reshape(-1).astype(bool)
                p0 = prev_pts[mask].reshape(-1, 2)
                p1 = curr_pts[mask].reshape(-1, 2)
                tracked = len(p0)
                if tracked >= 6:
                    m, inlier_mask = cv2.estimateAffinePartial2D(
                        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh,
                        confidence=0.99, refineIters=5
                    )
                    if m is not None:
                        rigid = normalize_to_rigid(m)
                        inliers = int(inlier_mask.sum()) if inlier_mask is not None else tracked

        pairwise.append(rigid)
        tracked_counts.append(tracked)
        inlier_ratios.append(inliers / tracked if tracked > 0 else 0.0)
        prev_gray = curr_gray

    cap.release()

    pairwise = np.stack(pairwise, axis=0)
    tracked_counts = np.asarray(tracked_counts)
    inlier_ratios = np.asarray(inlier_ratios)

    cumulative = [np.eye(3, dtype=np.float32)]
    for m in pairwise:
        cumulative.append(m @ cumulative[-1])
    cumulative = np.stack(cumulative, axis=0)
    trajectory = np.stack([params_from_mat(m) for m in cumulative], axis=0)

    return {
        "fps": fps,
        "width": w,
        "height": h,
        "pairwise": pairwise,
        "cumulative": cumulative,
        "trajectory": trajectory,
        "tracked_counts": tracked_counts,
        "inlier_ratios": inlier_ratios,
    }


def choose_crop(compensation: np.ndarray, width: int, height: int) -> tuple[int, int]:
    angles = np.arctan2(compensation[:, 1, 0], compensation[:, 0, 0])
    margin_x = np.abs(compensation[:, 0, 2]) + np.abs(np.sin(angles)) * (height / 2.0)
    margin_y = np.abs(compensation[:, 1, 2]) + np.abs(np.sin(angles)) * (width / 2.0)
    crop_x = int(min(max(np.percentile(margin_x, 97) + 10, 24), width * 0.12))
    crop_y = int(min(max(np.percentile(margin_y, 97) + 10, 24), height * 0.12))
    crop_x = min(crop_x, width // 5)
    crop_y = min(crop_y, height // 5)
    return crop_x, crop_y


def stabilize_video(video_path: str, compensation: np.ndarray, crop_x: int, crop_y: int, out_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame for warping")
    h, w = frame.shape[:2]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    while True:
        if idx >= len(compensation):
            break
        if idx > 0:
            ret, frame = cap.read()
            if not ret:
                break

        warped = cv2.warpAffine(frame, compensation[idx][:2], (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        cropped = warped[crop_y:h - crop_y, crop_x:w - crop_x]
        if cropped.size > 0:
            warped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(warped)
        idx += 1

    writer.release()
    cap.release()
    return idx


def put_text(img: np.ndarray, text: str, xy=(20, 50), scale=1.0) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def get_frame(video_path: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
    return frame


def build_comparison_video(orig_path: str, stab_path: str, out_path: str) -> None:
    cap1 = cv2.VideoCapture(orig_path)
    cap2 = cv2.VideoCapture(stab_path)
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        raise RuntimeError("Failed to read videos for comparison rendering")

    h, w = f1.shape[:2]
    target_h = 960 if h >= 960 else h
    scale = target_h / h
    target_w = int(w * scale)
    fps = float(cap1.get(cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w * 2, target_h))
    while ret1 and ret2:
        a = cv2.resize(f1, (target_w, target_h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(f2, (target_w, target_h), interpolation=cv2.INTER_AREA)
        a = put_text(a, "Original")
        b = put_text(b, "Stabilized")
        writer.write(np.concatenate([a, b], axis=1))
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()

    writer.release()
    cap1.release()
    cap2.release()


def build_figures(orig_path: str, stab_path: str, trajectory: np.ndarray, smoothed: np.ndarray,
                  compensation: np.ndarray, crop_x: int, crop_y: int, tracked: np.ndarray,
                  inlier: np.ndarray, fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    n_frames = min(len(trajectory), int(cv2.VideoCapture(stab_path).get(cv2.CAP_PROP_FRAME_COUNT)))

    t = np.arange(len(trajectory))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ["x, px", "y, px", "angle, rad"]
    for i, ax in enumerate(axes):
        ax.plot(t, trajectory[:, i], label="Original", linewidth=1.5)
        ax.plot(t, smoothed[:, i], label="Smoothed", linewidth=2.0)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Camera trajectory before/after smoothing")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "trajectory_plot.png"), dpi=150)
    plt.close(fig)

    sample_ids = [0, n_frames // 5, 2 * n_frames // 5, 3 * n_frames // 5, 4 * n_frames // 5, n_frames - 1]
    pairs = []
    for idx in sample_ids:
        a = put_text(get_frame(orig_path, idx), f"Original #{idx}")
        b = put_text(get_frame(stab_path, idx), f"Stabilized #{idx}")
        pairs.append(np.concatenate([a, b], axis=1))
    rows = []
    for i in range(0, len(pairs), 2):
        rows.append(np.concatenate(pairs[i:i + 2], axis=0))
    grid = np.concatenate(rows, axis=1)
    scale = 2200 / grid.shape[1]
    grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(fig_dir, "before_after_grid.png"), grid)

    score = np.where(tracked > 20, 1 - inlier, 0)
    hard_pairs = np.argsort(score)[-3:]
    warp_ids = sorted(set([int(x) for x in hard_pairs] + [n_frames // 2]))
    h, w = get_frame(orig_path, 0).shape[:2]
    rows = []
    for idx in warp_ids:
        fr = get_frame(orig_path, idx)
        warped = cv2.warpAffine(fr, compensation[idx][:2], (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        cropped = warped[crop_y:h - crop_y, crop_x:w - crop_x]
        final = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR) if cropped.size > 0 else warped
        row = np.concatenate([
            put_text(fr, f"Original #{idx}", (20, 40), 0.9),
            put_text(warped, "Warped + reflect", (20, 40), 0.9),
            put_text(final, "Final cropped", (20, 40), 0.9),
        ], axis=1)
        rows.append(row)
    canvas = np.concatenate(rows, axis=0)
    scale = 2500 / canvas.shape[1]
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(fig_dir, "warp_examples.png"), canvas)

    pair_idx = int(hard_pairs[-1]) if len(hard_pairs) > 0 else n_frames // 2
    cap = cv2.VideoCapture(orig_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pair_idx)
    ret, prev = cap.read()
    ret2, curr = cap.read()
    cap.release()
    if ret and ret2:
        scale = 0.25
        prev_small = cv2.resize(prev, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pg = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        cg = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(pg, maxCorners=220, qualityLevel=0.01, minDistance=10, blockSize=3)
        vis = curr_small.copy()
        if prev_pts is not None and len(prev_pts) >= 8:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                pg, cg, prev_pts, None, winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            mask = status.reshape(-1).astype(bool)
            p0 = prev_pts[mask].reshape(-1, 2)
            p1 = curr_pts[mask].reshape(-1, 2)
            _, inmask = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            if inmask is None:
                inmask = np.ones((len(p0), 1), dtype=np.uint8)
            for (x0, y0), (x1, y1), ok in zip(p0, p1, inmask.reshape(-1)):
                color = (0, 255, 0) if int(ok) else (0, 0, 255)
                cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)), color, 1, tipLength=0.25)
                cv2.circle(vis, (int(x1), int(y1)), 2, color, -1)
        vis = put_text(vis, f"LK tracks {pair_idx}->{pair_idx + 1}", (20, 28), 0.9)
        cv2.imwrite(os.path.join(fig_dir, "lk_tracks_example.png"), vis)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini camera stabilizer based on sparse Lucas–Kanade optical flow.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--out-dir", required=True, help="Directory for outputs")
    parser.add_argument("--scale", type=float, default=0.25, help="Downscale factor used only for motion estimation")
    parser.add_argument("--max-corners", type=int, default=220)
    parser.add_argument("--quality", type=float, default=0.01)
    parser.add_argument("--min-dist", type=int, default=10)
    parser.add_argument("--lk-win", type=int, default=15)
    parser.add_argument("--lk-levels", type=int, default=2)
    parser.add_argument("--ransac-thresh", type=float, default=2.5)
    parser.add_argument("--smooth-radius", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    fig_dir = out_dir / "figures"
    video_dir = out_dir / "output"
    src_dir = out_dir / "src"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    result = estimate_motion(
        args.input,
        scale=args.scale,
        max_corners=args.max_corners,
        quality=args.quality,
        min_dist=args.min_dist,
        lk_win=(args.lk_win, args.lk_win),
        lk_levels=args.lk_levels,
        ransac_thresh=args.ransac_thresh,
    )

    trajectory = result["trajectory"]
    smoothed = smooth_trajectory(trajectory, args.smooth_radius)
    smoothed_cumulative = np.stack([mat_from_params(*p) for p in smoothed], axis=0)
    compensation = np.stack([
        smoothed_cumulative[i] @ np.linalg.inv(result["cumulative"][i]) for i in range(len(result["cumulative"]))
    ], axis=0)
    crop_x, crop_y = choose_crop(compensation, result["width"], result["height"])

    stabilized_path = str(video_dir / "stabilized_lk.mp4")
    comparison_path = str(video_dir / "comparison_side_by_side.mp4")
    actual_frames = stabilize_video(args.input, compensation, crop_x, crop_y, stabilized_path)
    build_comparison_video(args.input, stabilized_path, comparison_path)
    build_figures(args.input, stabilized_path, trajectory[:actual_frames], smoothed[:actual_frames],
                  compensation[:actual_frames], crop_x, crop_y, result["tracked_counts"], result["inlier_ratios"],
                  str(fig_dir))

    np.save(data_dir / "trajectory.npy", trajectory)
    np.save(data_dir / "smoothed_trajectory.npy", smoothed)
    np.save(data_dir / "compensation.npy", compensation)

    metrics = {
        "fps": result["fps"],
        "width": result["width"],
        "height": result["height"],
        "n_frames_actual": actual_frames,
        "duration_sec": actual_frames / result["fps"],
        "crop_x_px": crop_x,
        "crop_y_px": crop_y,
        "tracked_mean": float(np.mean(result["tracked_counts"])),
        "tracked_median": float(np.median(result["tracked_counts"])),
        "inlier_ratio_mean": float(np.mean(result["inlier_ratios"])),
        "inlier_ratio_median": float(np.median(result["inlier_ratios"])),
        "params": vars(args),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
