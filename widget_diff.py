# pip install torch torchvision ftfy regex tqdm git+https://github.com/openai/CLIP.git
import cv2, torch, clip
import numpy as np
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _extract_boxes_and_diff(img_a, img_b, threshold=0.3, min_area=50):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    _, sim_map = ssim(gray_a, gray_b, full=True)
    diff_u8 = ((1.0 - sim_map) * 255.0).astype("uint8")
    _, mask = cv2.threshold(diff_u8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
    return boxes, diff_u8, mask

def _crop(img, box, pad=8):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
    return img[y0:y1, x0:x1], (x0, y0, x1-x0, y1-y0)

def _upscale_min_side(img_bgr, min_side=256):
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    if s >= min_side:
        return img_bgr
    scale = float(min_side) / float(s)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_CUBIC)

def _load_clip(model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=DEVICE, jit=False)
    model.eval()
    return model, preprocess

def _embed_images(model, preprocess, imgs_bgr: List[np.ndarray]) -> torch.Tensor:
    pil = []
    for im in imgs_bgr:
        im = _upscale_min_side(im, min_side=256)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pil.append(preprocess(Image.fromarray(rgb)))
    batch = torch.stack(pil).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def _embed_texts(model, texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        tok = clip.tokenize(texts).to(DEVICE)
        feats = model.encode_text(tok)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def localize_and_label_changes(
    img_a_path: str,
    img_b_path: str,
    ssim_threshold: float = 0.3,
    min_area: int = 50,
    change_sim_threshold: float = 0.95,
    label_sim_threshold: float = 0.23,
    # Descriptive prompts for CLIP
    prompt_labels: List[str] = (
        "a mobile toggle switch",
        "a settings screen toggle",
        "a rectangular mobile UI button",
        "a text input field in a form",
        "a checkbox control in a list",
        "a small monochrome app icon",
        "an image view showing a picture",
        "a text link styled as hyperlink",
        "a plain text label in a settings row",
        "an unknown ui element"
    ),
    # Map descriptive -> concise display name
    display_map: Dict[str, str] = None,
    model_name: str = "ViT-B/32",
):
    if display_map is None:
        display_map = {
            "a mobile toggle switch": "toggle",
            "a settings screen toggle": "toggle",
            "a rectangular mobile UI button": "button",
            "a text input field in a form": "input box",
            "a checkbox control in a list": "checkbox",
            "a small monochrome app icon": "icon",
            "an image view showing a picture": "image",
            "a text link styled as hyperlink": "link",
            "a plain text label in a settings row": "text",
            "an unknown ui element": "unknown",
        }

    img_a = cv2.imread(img_a_path)  # new
    img_b = cv2.imread(img_b_path)  # reference
    if img_a is None or img_b is None:
        raise ValueError("Could not read images")

    if img_a.shape[:2] != img_b.shape[:2]:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)

    boxes, diff_u8, mask = _extract_boxes_and_diff(img_a, img_b, threshold=ssim_threshold, min_area=min_area)
    if not boxes:
        return {"boxes": [], "results": [], "overlay_rgb": cv2.cvtColor(img_a.copy(), cv2.COLOR_BGR2RGB), "diff_u8": diff_u8, "mask": mask}

    crops_new, crops_ref, out_boxes = [], [], []
    for b in boxes:
        ca, bb = _crop(img_a, b, pad=8)
        cb, _  = _crop(img_b, b, pad=8)
        if ca.size == 0 or cb.size == 0:
            continue
        crops_new.append(ca); crops_ref.append(cb); out_boxes.append(bb)

    model, preprocess = _load_clip(model_name)
    feats_new = _embed_images(model, preprocess, crops_new)
    feats_ref = _embed_images(model, preprocess, crops_ref)

    sim_img = (feats_new * feats_ref).sum(dim=-1)
    is_changed = sim_img < change_sim_threshold

    text_emb = _embed_texts(model, list(prompt_labels))

    results = []
    for i, box in enumerate(out_boxes):
        sims = feats_new[i] @ text_emb.T
        top_idx = int(sims.argmax().item())
        top_val = float(sims[top_idx].item())

        if top_val >= label_sim_threshold:
            raw_label = prompt_labels[top_idx]
            display_label = display_map.get(raw_label, raw_label)
        else:
            raw_label = "an unknown ui element"
            display_label = "unknown"

        rec = {
            "box": box,
            "sim_ref": float(sim_img[i]),
            "changed": bool(is_changed[i]),
            "label": display_label,     # concise
            "label_sim": top_val,       # similarity to the descriptive prompt
            "raw_label": raw_label      # verbose prompt that matched
        }
        results.append(rec)

    overlay = cv2.cvtColor(img_a.copy(), cv2.COLOR_BGR2RGB)
    for r in results:
        x, y, w, h = r["box"]
        color = (0, 255, 0) if r["changed"] else (255, 215, 0)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
        if r["changed"]:
            txt = f'{r["label"]} ({r["label_sim"]:.2f})'
            cv2.putText(overlay, txt, (x, max(0, y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return {
        "boxes": out_boxes,
        "results": results,
        "overlay_rgb": overlay,
        "diff_u8": diff_u8,
        "mask": mask
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    result = localize_and_label_changes("settings_1.png", "settings_2.png")

    for r in result["results"]:
        print(r)

    plt.figure(figsize=(10, 8))
    plt.imshow(result["overlay_rgb"]); plt.axis("off"); plt.title("Detected UI Changes")
    plt.show()
