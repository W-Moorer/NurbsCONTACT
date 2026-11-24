"""Orthographic three-dimensional playback of mesh solids at the recorded poses."""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


def read_triangles(path):
    import meshio

    data = meshio.read(path)
    return data.points[data.get_cells_type("triangle")]


def render_solids(video, triangles, translation, angles, time, loads, title):
    """Angles are signed world-Z rotations. All lengths use one common unit."""
    time = np.asarray(time)
    translation = np.broadcast_to(translation, (len(time), len(triangles), 3))
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("Expected an increasing physical time axis")
    indices = np.unique(np.rint(np.linspace(0, len(time) - 1, min(len(time), 251))).astype(int))
    plot = pv.Plotter(off_screen=True, window_size=(960, 464))
    plot.set_background("#f2f5f7")
    colors = ("#327895", "#759fb1", "#c48346", "#e1b17a")
    actors = []
    bounds = []
    for j, tri in enumerate(triangles):
        vertices, indices_flat = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
        faces = np.column_stack((np.full(len(tri), 3), indices_flat.reshape(-1, 3)))
        actor = plot.add_mesh(
            pv.PolyData(vertices, faces.ravel()),
            color=colors[j % len(colors)],
            smooth_shading=True,
            split_sharp_edges=True,
            ambient=0.26,
            specular=0.22,
        )
        actors.append(actor)
        r = np.linalg.norm(vertices[:, :2], axis=1).max()
        bounds.extend(
            (
                translation[:, j].min(0) + [-r, -r, vertices[:, 2].min()],
                translation[:, j].max(0) + [r, r, vertices[:, 2].max()],
            )
        )
    lo, hi = np.min(bounds, axis=0), np.max(bounds, axis=0)
    center = 0.5 * (lo + hi)
    direction = np.array([0.30, -0.97, 0.89])
    direction /= np.linalg.norm(direction)
    right = np.cross([0, 0, 1], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    corners = np.array(np.meshgrid(*zip(lo, hi))).reshape(3, -1).T
    plot.camera_position = (center + 3 * np.linalg.norm(hi - lo) * direction, center, up)
    plot.enable_parallel_projection()
    plot.camera.parallel_scale = 0.51 * max(
        np.ptp(corners @ up), np.ptp(corners @ right) * 464 / 960
    )
    font_name = "arial.ttf" if Path("C:/Windows/Fonts/arial.ttf").exists() else "DejaVuSans.ttf"
    font = ImageFont.truetype(font_name, 21)
    video = Path(video)
    temporary = video.with_name(video.stem + ".partial.mp4")
    try:
        with imageio.get_writer(
            temporary,
            fps=25,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p", "-threads", "2"],
            ffmpeg_log_level="error",
        ) as writer:
            for i in indices:
                for j, actor in enumerate(actors):
                    actor.position = translation[i, j]
                    actor.orientation = (0, 0, np.degrees(angles[i, j] % (2 * np.pi)))
                plot.renderer.ResetCameraClippingRange()
                plot.render()
                img = Image.new("RGB", (960, 544), "#f2f5f7")
                img.paste(Image.fromarray(plot.screenshot(return_img=True)), (0, 39))
                draw = ImageDraw.Draw(img)
                draw.text((22, 8), title, fill="#233e4e", font=font)
                draw.text((938, 8), f"{time[i]:.4f} s", anchor="ra", fill="#233e4e", font=font)
                values = (
                    "Recorded joint angles"
                    if loads is None
                    else "     ".join(
                        f"Pair {k+1}   {f/1000:.3f} kN" for k, f in enumerate(loads[i])
                    )
                )
                draw.text((22, 505), values, fill="#294858", font=font)
                draw.rectangle(
                    (0, 539, int(959 * (time[i] - time[0]) / (time[-1] - time[0])), 543),
                    fill="#327895",
                )
                writer.append_data(np.asarray(img))
        temporary.replace(video)
    finally:
        plot.close()
        temporary.unlink(missing_ok=True)


def render_reducer_animation(deck, video, angle, force, t, title="Two-stage reducer"):
    with np.load(deck / "assembly.npz") as model:
        bodies, mapping = model["bodies"], model["mapping"]
    triangles = []
    for part in json.loads((deck / "models.json").read_text())["parts"]:
        with np.load(deck / part["archive"]) as data:
            triangles.append(data["tri"])
    render_solids(
        video,
        triangles,
        bodies[:, 4:7],
        bodies[:, 3] * (angle[:, mapping] + bodies[:, 7]),
        t,
        force,
        title,
    )
