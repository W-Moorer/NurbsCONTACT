"""Verify the rendered artifacts and make Python contact sheets for visual QA."""

import argparse
import json
from pathlib import Path

import av
import fitz
import numpy as np
from PIL import Image, ImageDraw

from paper import paper_output_path


def sheet(items, size, columns, path):
    rows = (len(items) + columns - 1) // columns
    width, height = size
    canvas = Image.new("RGB", (width*columns, (height+23)*rows), "#e4eaf0")
    labels = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        x, y = width*(index % columns), (height+23)*(index//columns)
        resized = image.copy()
        resized.thumbnail(size)
        canvas.paste(resized, (x+(width-resized.width)//2, y))
        labels.text((x+8, y+height+3), label, fill="#142b40")
    canvas.save(path)


def verify_video(root, entry):
    path = root/"animations"/entry["file"]
    targets = sorted({0, entry["frames"]//3, 2*entry["frames"]//3, entry["frames"]-1})
    samples = []
    times = []
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        assert video.width == entry["width"] and video.height == entry["height"], path
        assert video.average_rate == entry["fps"], path
        frames = 0
        for index, frame in enumerate(container.decode(video)):
            frames += 1
            if index in targets:
                samples.append(frame.to_image())
                times.append(frame.time)
        assert frames == entry["frames"], path
    assert np.all(np.diff(times) > 0), path
    content = np.stack([np.asarray(image, dtype=float)[83:-9, :, :] for image in samples])
    changed = np.mean(np.abs(content[1:] - content[0:1]), axis=(1, 2, 3))
    assert max(changed) > 0.1, f"No visible modeled motion in {path}"
    return samples, {"file": path.name, "frames": frames, "max_mean_pixel_change": float(max(changed))}


def verify_paper(root, qa):
    assets = ("model-scenes", "calibration", "cost-accuracy", "long-run")
    panels = []
    for name in assets:
        with fitz.open(root/"figures"/f"{name}.pdf") as figure:
            assert len(figure) == 1, name
            with Image.open(root/"figures"/f"{name}.png") as png:
                panels.append((name, png.copy()))
    sheet(panels, (620, 390), 2, qa/"figures.png")

    with fitz.open(paper_output_path(root)) as paper:
        previews = []
        pages = []
        for index, page in enumerate(paper, start=1):
            text = page.get_text()
            assert text.strip() and "??" not in text, f"Paper page {index}"
            pages.append(text)
            pixels = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
            image = Image.frombytes("RGB", (pixels.width, pixels.height), pixels.samples)
            image.save(qa/f"paper-{index}.png")
            previews.append((f"Page {index}", image))
        for required in ("References", "Reproducibility", "OpenAMRobot", "Figure 4"):
            assert required in "\n".join(pages), f"Missing {required} in PDF"
    sheet(previews, (298, 421), 3, qa/"paper.png")
    return len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("previews", type=Path)
    args = parser.parse_args()
    root, qa = args.results, args.previews
    qa.mkdir(exist_ok=True, parents=True)
    entries = json.loads((root/"animations"/"index.json").read_text(encoding="utf-8"))
    assert len(entries) == 6 and len({e["file"] for e in entries}) == 6
    items = []
    videos = []
    views = {}
    for entry in entries:
        frames, result = verify_video(root, entry)
        videos.append(result)
        scene = entry["scene"]
        views.setdefault(scene, []).append(np.asarray(frames[0], dtype=float)[83:-9])
        for name, image in zip(("start", "one third", "two thirds", "end"), frames, strict=True):
            items.append((f"{scene} / {entry['view']} / {name}", image))
    for scene, frames in views.items():
        assert len(frames) == 2 and np.mean(np.abs(frames[0]-frames[1])) > 0.25, scene
    sheet(items, (310, 174), 4, qa/"animations.png")
    stats = {"paper_pages": verify_paper(root, qa), "videos": videos}
    (qa/"verification.json").write_text(json.dumps(stats, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
