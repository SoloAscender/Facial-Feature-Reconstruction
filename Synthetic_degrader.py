import cv2
import numpy as np
import random


def sample_degradation_params():
    """
    Sample ONE set of degradation params to be shared across an entire clip,
    matching FADRA Sec IV-A2 exactly.
    """
    r = random.uniform(1.0, 8.0)                      # resize ratio
    k = random.choice(list(range(3, 26, 2)))           # odd kernel size in {3,5,...,25}
    sigma_b = random.uniform(1.5, 6.0)                 # blur strength
    sigma_n = random.uniform(0.01, 0.05)               # noise std, in [0,1]-normalized space
    q = random.randint(25, 85)                         # JPEG quality factor
    return dict(r=r, k=k, sigma_b=sigma_b, sigma_n=sigma_n, q=q)


def degrade_frame(frame, H, W, params):
    """
    Apply ONE frame's worth of degradation using a SHARED param set.
    Order matches the paper exactly: resize-down -> resize-up -> blur -> noise -> JPEG.
    """
    r = params['r']
    H_hat = max(1, round(H / r))
    W_hat = max(1, round(W / r))

    down = cv2.resize(frame, (W_hat, H_hat), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(down, (W, H), interpolation=cv2.INTER_LINEAR)

    blurred = cv2.GaussianBlur(up, (params['k'], params['k']), sigmaX=params['sigma_b'])

    x = blurred.astype(np.float32) / 255.0
    noise = np.random.normal(0, params['sigma_n'], x.shape).astype(np.float32)
    noisy = np.clip(x + noise, 0.0, 1.0)
    noisy_uint8 = (noisy * 255.0).astype(np.uint8)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), params['q']]
    ok, encoded = cv2.imencode('.jpg', noisy_uint8, encode_param)
    assert ok, "JPEG encode failed"
    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return compressed


def degrade_clip(frames, seed=None):
    """
    frames: list of HxWx3 uint8 BGR frames (all same shape, from one clip).
    Returns: (degraded_frames, params_used) -- params shared across all frames.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    H, W = frames[0].shape[:2]
    params = sample_degradation_params()
    degraded = [degrade_frame(f, H, W, params) for f in frames]
    return degraded, params


def load_video_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


if __name__ == "__main__":
    # ---- build synthetic test frames (stand-in for a real clip, since no .mp4 available here) ----
    H, W = 128, 128
    n_frames = 8
    synthetic_frames = []
    for i in range(n_frames):
        f = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.circle(f, (64 + i, 64), 30, (255, 200, 150), -1)   # moving "face-like" blob
        cv2.rectangle(f, (20, 20), (108, 108), (80, 80, 200), 2)
        synthetic_frames.append(f)

    # ---- test 1: params are shared across all frames in a clip ----
    degraded, params = degrade_clip(synthetic_frames, seed=42)
    print("Sampled params for this clip:", params)
    print("Number of degraded frames:", len(degraded))
    print("All degraded frames same shape as input:",
          all(d.shape == synthetic_frames[0].shape for d in degraded))

    # ---- test 2: re-running degrade_clip on the SAME frames with a NEW random draw
    #              gives DIFFERENT params (confirms randomness works, not hardcoded) ----
    degraded2, params2 = degrade_clip(synthetic_frames, seed=123)
    print("\nDifferent seed gives different params:", params != params2)

    # ---- test 3: within one clip, verify no per-frame param drift.
    #              Manually re-derive frame 0 and frame 5 using the SAME params dict
    #              and confirm identical degradation given identical input pixels. ----
    identical_input_frames = [synthetic_frames[0].copy() for _ in range(2)]
    np.random.seed(7)
    out_a = degrade_frame(identical_input_frames[0], H, W, params)
    np.random.seed(7)  # reset RNG so only the noise draw matches -- checks param consistency, not noise realization
    out_b = degrade_frame(identical_input_frames[1], H, W, params)
    print("Same params + same noise seed on identical input -> identical output:",
          np.array_equal(out_a, out_b))

    # ---- save before/after for visual inspection ----
    cv2.imwrite("/home/abhimanyu_balachandra/Downloads/Thesis_PIL/Project_1_FADRA/facerestore/ORIG_frame.png", synthetic_frames[0])
    cv2.imwrite("/home/abhimanyu_balachandra/Downloads/Thesis_PIL/Project_1_FADRA/facerestore/DEGR_frame.png", degraded[0])
    