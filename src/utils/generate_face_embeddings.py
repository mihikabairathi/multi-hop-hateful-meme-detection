# generate_face_embeddings.py
#
# Build rich face embeddings for memes by combining:
#   - InsightFace (ArcFace) identity embeddings
#   - FER (research-grade facial expression model) emotion probabilities
#   - OpenFace 3.0 AUs + gaze
#
# The result is stored as "face_feats" in the existing CLIP .pt files:
#   data/CLIP_Embedding/FB/{split}_{clip_model_name}.pt
#
# Assumes:
#   - CLIP .pt files already exist from generate_clip_embeddings.py
#   - InsightFace, FER, and OpenFace 3.0 are installed
#   - OpenFace's RetinaFace weights are available in ./weights

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------
# imports for face libraries
# ---------------------------
from insightface.app import FaceAnalysis          # identity (ArcFace)
from openface.face_detection import FaceDetector  # gaze + AU
from openface.multitask_model import MultitaskPredictor

# Try to import FER from whatever layout this version of `fer` uses
FER = None
try:
    from fer import FER as _FER  # most versions: fer.FER
    FER = _FER
except Exception:
    try:
        from fer.fer import FER as _FER  # some versions: fer.fer.FER
        FER = _FER
    except Exception:
        FER = None  # will handle in build_emotion_model()

# ---------------------------
# import dataset utilities
# ---------------------------
try:
    from utils.dataset import get_values_from_gt
except ImportError:
    from dataset import get_values_from_gt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate rich face embeddings and add them to CLIP feature files.")
    parser.add_argument(
        "--EXP_FOLDER",
        type=str,
        default="./data/CLIP_Embedding",
        help="Folder where CLIP embedding .pt files live.",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai_clip-vit-large-patch14-336_HF",
        help="Suffix of CLIP feature files, e.g., train_<clip_model_name>.pt",
    )
    # OpenFace 3.0
    parser.add_argument(
        "--face_model_path",
        type=str,
        default="./weights/Alignment_RetinaFace.pth",
        help="Path to OpenFace RetinaFace detector weights.",
    )
    parser.add_argument(
        "--multitask_model_path",
        type=str,
        default="./weights/MTL_backbone.pth",
        help="Path to OpenFace multitask backbone weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for face models.",
    )
    parser.add_argument(
        "--insightface_detector_size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Detection size for InsightFace (width height).",
    )
    parser.add_argument(
        "--openface_resize",
        type=float,
        default=1.0,
        help="Resize factor for OpenFace FaceDetector.get_face.",
    )
    return parser.parse_args()


# ---------------------------
# Build models
# ---------------------------

def build_insightface(args):
    """
    ArcFace identity embedding via InsightFace (buffalo_l pack).
    """
    app = FaceAnalysis(name="buffalo_l")
    ctx_id = 0 if args.device == "cuda" else -1
    app.prepare(ctx_id=ctx_id, det_size=tuple(args.insightface_detector_size))
    return app


def build_emotion_model():
    """
    FER emotion model (Justin Shenk's library, trained on FER datasets).
    Uses internal CNN backbone; no TF.
    """
    if FER is None:
        raise RuntimeError(
            "Could not import FER class from the 'fer' package.\n"
            "Tried 'from fer import FER' and 'from fer.fer import FER'.\n"
            "Check that you have the correct 'fer' library installed, e.g.\n"
            "  pip install --upgrade 'fer==22.4.0'\n"
        )
    # mtcnn=True uses a decent face detector if available, else falls back.
    fer_model = FER(mtcnn=True)
    return fer_model


def build_openface(args):
    """
    OpenFace 3.0 FaceDetector + MultitaskPredictor (AUs + gaze).
    """
    detector = FaceDetector(
        model_path=args.face_model_path,
        device=args.device,
    )
    multitask = MultitaskPredictor(
        model_path=args.multitask_model_path,
        device=args.device,
    )
    return detector, multitask


# ---------------------------
# Per-component embeddings
# ---------------------------

def compute_identity_embedding_insightface(img_path, app):
    """
    Returns ArcFace identity embedding (normalized) or None.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None

    faces = app.get(bgr)
    if len(faces) == 0:
        return None

    # Choose the largest face (by area)
    def face_area(f):
        x1, y1, x2, y2 = f.bbox
        return float(x2 - x1) * float(y2 - y1)

    face = max(faces, key=face_area)
    emb = face.normed_embedding  # 512-D
    if emb is None:
        return None

    return emb.astype(np.float32)


EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def compute_emotion_embedding_fer(img_path, emo_model):
    """
    Returns emotion probability vector in EMOTION_ORDER using fer, or None.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    try:
        # detect_emotions returns a list of detections
        # each with {'box': [...], 'emotions': {...}}
        result = emo_model.detect_emotions(img)
    except Exception:
        return None

    if not result:
        return None

    # pick the largest face detection by box area
    def box_area(det):
        x, y, w, h = det["box"]
        return float(w) * float(h)

    best = max(result, key=box_area)
    emo_dict = best.get("emotions", None)
    if emo_dict is None:
        return None

    # FER keys usually are: angry, disgust, fear, happy, sad, surprise, neutral
    vals = np.array([emo_dict.get(k, 0.0) for k in EMOTION_ORDER], dtype=np.float32)
    s = vals.sum()
    if s > 1e-6:
        vals /= s
    return vals

def compute_openface_embedding(img_path, detector, multitask, resize=1.0):
    """
    Returns concatenated [gaze_vec, au_vec] from OpenFace 3.0, or None.
    Handles cases where the detector returns an empty crop.
    """
    cropped_face, dets = detector.get_face(img_path, resize=resize)

    # No detection or bad crop
    if cropped_face is None or dets is None:
        return None

    # Sometimes an empty array slips through; guard that too
    if not isinstance(cropped_face, np.ndarray) or cropped_face.size == 0:
        return None

    try:
        with torch.no_grad():
            emotion_logits, gaze_output, au_output = multitask.predict(cropped_face)
    except cv2.error:
        # OpenCV failed internally (e.g., cvtColor on empty), treat as no face
        return None
    except Exception:
        # Any other weirdness from OpenFace: just skip this sample
        return None

    gaze_vec = gaze_output.squeeze(0).cpu().numpy().astype(np.float32)  # [2]
    au_vec = au_output.squeeze(0).cpu().numpy().astype(np.float32)      # [n_AU]
    return np.concatenate([gaze_vec, au_vec], axis=0)


# ---------------------------
# Split processing
# ---------------------------

def process_split(split_name, args, app_insight, emo_model, of_detector, of_multitask):
    """
    For a given split:
      * read image paths using get_values_from_gt
      * compute identity, emotion, and OpenFace embeddings
      * infer per-component dims
      * build final concatenated face_feats
      * inject into CLIP .pt file as 'face_feats'
    """
    img_paths, texts, labels, ids = get_values_from_gt(split_name)

    component_list = []

    print(f"[{split_name}] computing face components...")
    for img_path in tqdm(img_paths):
        comps = {}

        # 1) identity
        id_vec = compute_identity_embedding_insightface(img_path, app_insight)
        comps["id"] = id_vec

        # 2) emotion (FER)
        emo_vec = compute_emotion_embedding_fer(img_path, emo_model)
        comps["emo"] = emo_vec

        # 3) openface (gaze + AU)
        of_vec = compute_openface_embedding(img_path, of_detector, of_multitask, resize=args.openface_resize)
        comps["of"] = of_vec

        component_list.append(comps)

    # infer dims from first non-None example per component
    def infer_dim(key):
        for c in component_list:
            if c[key] is not None:
                return c[key].shape[0]
        return 0

    id_dim = infer_dim("id")
    emo_dim = infer_dim("emo")
    of_dim = infer_dim("of")

    if id_dim == emo_dim == of_dim == 0:
        raise RuntimeError(f"No face components could be extracted for split '{split_name}'.")

    print(f"[{split_name}] dims -> id: {id_dim}, emotion: {emo_dim}, openface: {of_dim}")

    # second pass: build final concatenated embeddings
    final_vecs = []
    for comps in component_list:
        parts = []
        # identity
        if id_dim > 0:
            parts.append(comps["id"] if comps["id"] is not None else np.zeros(id_dim, dtype=np.float32))
        # emotion
        if emo_dim > 0:
            parts.append(comps["emo"] if comps["emo"] is not None else np.zeros(emo_dim, dtype=np.float32))
        # openface gaze + AU
        if of_dim > 0:
            parts.append(comps["of"] if comps["of"] is not None else np.zeros(of_dim, dtype=np.float32))

        final_vecs.append(np.concatenate(parts, axis=0))

    face_feats = torch.from_numpy(np.stack(final_vecs, axis=0))  # [N, D_total]

    # ---------------------------
    # Inject into CLIP .pt file
    # ---------------------------
    pt_path = os.path.join(args.EXP_FOLDER, "FB", f"{split_name}_{args.clip_model_name}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"Could not find CLIP feature file for split '{split_name}': {pt_path}. "
            f"Make sure you've run generate_clip_embeddings.py first."
        )

    data = torch.load(pt_path, map_location="cpu")
    num_existing = data["img_feats"].shape[0]
    if num_existing != face_feats.shape[0]:
        raise ValueError(
            f"Sample count mismatch for split '{split_name}': "
            f"CLIP feats have {num_existing}, face_feats have {face_feats.shape[0]}"
        )

    data["face_feats"] = face_feats
    torch.save(data, pt_path)
    print(f"[{split_name}] saved updated file with face_feats: {pt_path}")


def main():
    args = parse_args()

    # Make sure repo root is on sys.path for dataset imports
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    print("Building InsightFace (ArcFace) model...")
    app_insight = build_insightface(args)

    print("Building FER Emotion model (research-grade)...")
    emo_model = build_emotion_model()

    print("Building OpenFace 3.0 models...")
    of_detector, of_multitask = build_openface(args)

    for split in ["train", "dev_seen", "test_seen", "test_unseen"]:
        process_split(split, args, app_insight, emo_model, of_detector, of_multitask)


if __name__ == "__main__":
    main()
