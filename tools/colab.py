"""Google Colab helpers: move the big sample videos Drive -> Drive -> local disk, and a
click tool to draw the scene map on a frame. Only imported inside Colab (dev tooling).

Why server-side copy: public "anyone with the link" files of several GB often fail with
"Too many users have viewed or downloaded this file" when downloaded directly. files.copy
makes a copy inside YOUR Drive without downloading anything; Colab then reads it through
the Drive mount at internal-network speed.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import time

import cv2
import numpy as np

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
_SVC = {}


# ============================================================ Drive
def drive_service():
    if "svc" not in _SVC:
        from google.colab import auth
        from googleapiclient.discovery import build
        auth.authenticate_user()
        _SVC["svc"] = build("drive", "v3", cache_discovery=False)
    return _SVC["svc"]


def file_id(link: str) -> str:
    m = re.search(r"/d/([\w-]{20,})", link) or re.search(r"[?&]id=([\w-]{20,})", link)
    if not m:
        raise ValueError(f"cannot find a file id in {link}")
    return m.group(1)


def resolve_folder(mydrive_rel: str) -> str:
    """Folder id for a path relative to My Drive (creates folders, follows shortcuts)."""
    svc = drive_service()
    parent = "root"
    for name in [p for p in mydrive_rel.strip("/").split("/") if p]:
        q = f"name = '{name}' and '{parent}' in parents and trashed = false"
        hits = svc.files().list(q=q, fields="files(id, mimeType, shortcutDetails)", spaces="drive",
                                supportsAllDrives=True, includeItemsFromAllDrives=True).execute()["files"]
        folder = next((h for h in hits if h["mimeType"] == FOLDER_MIME), None)
        short = next((h for h in hits if h["mimeType"] == SHORTCUT_MIME), None)
        if folder:
            parent = folder["id"]
        elif short:
            parent = short["shortcutDetails"]["targetId"]
        else:
            parent = svc.files().create(body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent]},
                                        fields="id").execute()["id"]
    return parent


def _wait_visible(path: str, size: int, timeout: float = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path) and os.path.getsize(path) == size:
            return True
        time.sleep(5)
    return False


def import_videos(links: list[str], drive_dir: str, mount_root: str = "/content/drive/MyDrive") -> list[str]:
    """Server-side copy each shared video into <My Drive>/<drive_dir>/ (skips ones already there).
    Returns the paths of the videos as seen through the Drive mount."""
    svc = drive_service()
    folder = None
    paths = []
    for link in links:
        fid = file_id(link)
        meta = svc.files().get(fileId=fid, fields="id, name, size, md5Checksum",
                               supportsAllDrives=True).execute()
        name, size = meta["name"], int(meta.get("size", 0))
        dst = os.path.join(mount_root, drive_dir, name)
        paths.append(dst)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            print(f"✓ {name} already in Drive ({size / 2**30:.2f} GB)")
            continue
        folder = folder or resolve_folder(drive_dir)
        print(f"→ copying {name} ({size / 2**30:.2f} GB) inside Google Drive (no download)...", flush=True)
        t0 = time.time()
        try:
            svc.files().copy(fileId=fid, body={"name": name, "parents": [folder]},
                             supportsAllDrives=True, fields="id").execute()
        except Exception as exc:  # timeouts on huge files: the copy usually still completes
            print("   copy call returned an error, waiting to see if it lands anyway:", str(exc)[:200])
        if not _wait_visible(dst, size):
            from google.colab import drive
            drive.flush_and_unmount()
            drive.mount("/content/drive", force_remount=True)
            if not _wait_visible(dst, size, 120):
                print(f"   ✗ {name} not visible yet. Fallbacks: download_with_gdown(), or open the link → "
                      f"'Add shortcut to Drive' → put it in My Drive/{drive_dir}")
                continue
        print(f"   ✓ done in {time.time() - t0:.0f}s")
    return paths


def download_with_gdown(links: list[str], local_dir: str) -> list[str]:
    """Fallback: direct download to the Colab disk (may hit Google's download quota)."""
    import gdown
    os.makedirs(local_dir, exist_ok=True)
    out = []
    for link in links:
        p = gdown.download(id=file_id(link), output=local_dir + "/", quiet=False)
        if p:
            out.append(p)
    return out


def _lower_ext(name: str) -> str:
    """C3896.MP4 -> C3896.mp4 (Linux globs are case-sensitive; tools and harness look for *.mp4)."""
    stem, ext = os.path.splitext(name)
    return stem + ext.lower()


def download_via_api(links: list[str], local_dir: str, chunk_mb: int = 256) -> list[str]:
    """Authenticated Drive API download straight to the Colab disk. Nothing is stored in your
    Drive (use when your Drive is full). Not subject to the public-link download quota."""
    import io
    from googleapiclient.http import MediaIoBaseDownload
    svc = drive_service()
    os.makedirs(local_dir, exist_ok=True)
    out = []
    for link in links:
        fid = file_id(link)
        meta = svc.files().get(fileId=fid, fields="name, size", supportsAllDrives=True).execute()
        dst = os.path.join(local_dir, _lower_ext(meta["name"]))
        out.append(dst)
        if os.path.exists(dst) and os.path.getsize(dst) == int(meta["size"]):
            print(f"✓ {os.path.basename(dst)} already on local disk"); continue
        t0 = time.time()
        req = svc.files().get_media(fileId=fid, supportsAllDrives=True)
        with io.FileIO(dst, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=chunk_mb * 2**20)
            done = False
            while not done:
                status, done = dl.next_chunk()
                print(f"\r{meta['name']}: {status.progress() * 100:5.1f}%", end="", flush=True)
        print(f"  done in {time.time() - t0:.0f}s")
    return out


def copy_local(src_paths: list[str], local_dir: str) -> list[str]:
    """Drive mount -> local SSD once per session (fast random access for decoding)."""
    os.makedirs(local_dir, exist_ok=True)
    out = []
    for src in src_paths:
        if not os.path.exists(src):
            continue
        dst = os.path.join(local_dir, _lower_ext(os.path.basename(src)))
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
            t0 = time.time()
            print(f"copying {os.path.basename(src)} to local disk...", end=" ", flush=True)
            shutil.copyfile(src, dst)
            print(f"{os.path.getsize(dst) / 2**20 / max(time.time() - t0, 1e-3):.0f} MB/s")
        out.append(dst)
    return out


def verify(paths: list[str]) -> None:
    """ffprobe each file: a truncated copy shows up as a wrong duration or an error."""
    for p in paths:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,r_frame_rate,codec_name",
                            "-of", "json", p], capture_output=True, text=True)
        if r.returncode:
            print("✗", os.path.basename(p), r.stderr[:200]); continue
        j = json.loads(r.stdout)
        s = j["streams"][0]; d = float(j["format"]["duration"])
        print(f"✓ {os.path.basename(p):40s} {int(d // 60)}:{d % 60:04.1f}  {s['width']}x{s['height']}  "
              f"{s['r_frame_rate']} fps  {s['codec_name']}  {os.path.getsize(p) / 2**30:.2f} GB")


# ============================================================ display + click tool
def show(img: np.ndarray, width: int = 1100):
    from IPython.display import Image, display
    s = min(1.0, width / img.shape[1])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    display(Image(data=cv2.imencode(".jpg", small)[1].tobytes()))


_JS = r"""
async function wiutPick(b64, w, h, title, mode) {
  const div = document.createElement('div');
  div.style.cssText = 'border:1px solid #999;padding:6px;margin:4px 0;font-family:sans-serif';
  const hint = {polygon:'click the corners, then Done', polyline:'click along the line, then Done',
                line:'click 2 points (for a direction: start then end)', box:'click 2 opposite corners'}[mode];
  div.innerHTML = '<b>' + title + '</b> - ' + hint + '<br>';
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  cv.style.cssText = 'cursor:crosshair;max-width:100%';
  const undo = document.createElement('button'); undo.textContent = 'Undo';
  const done = document.createElement('button'); done.textContent = 'Done'; done.style.marginLeft = '8px';
  div.appendChild(cv); div.appendChild(document.createElement('br')); div.appendChild(undo); div.appendChild(done);
  document.body.appendChild(div);
  const ctx = cv.getContext('2d'); const img = new Image(); const pts = [];
  const redraw = () => {
    ctx.drawImage(img, 0, 0, w, h); ctx.lineWidth = 2; ctx.strokeStyle = '#ff0'; ctx.fillStyle = '#f00';
    if (mode === 'box' && pts.length === 2) {
      ctx.strokeRect(pts[0][0], pts[0][1], pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
    } else if (pts.length) {
      ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
      if (mode === 'polygon' && pts.length > 2) ctx.closePath(); ctx.stroke();
      if (mode === 'line' && pts.length === 2) { ctx.beginPath(); ctx.arc(pts[1][0], pts[1][1], 8, 0, 7); ctx.stroke(); }
    }
    pts.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 7); ctx.fill(); });
  };
  await new Promise(r => { img.onload = r; img.src = 'data:image/jpeg;base64,' + b64; });
  redraw();
  cv.onclick = e => {
    const r = cv.getBoundingClientRect();
    if ((mode === 'line' || mode === 'box') && pts.length >= 2) return;
    pts.push([(e.clientX - r.left) * w / r.width, (e.clientY - r.top) * h / r.height]); redraw();
  };
  undo.onclick = () => { pts.pop(); redraw(); };
  await new Promise(r => done.onclick = r);
  div.innerHTML = '<i>' + title + ': ' + pts.length + ' points saved</i>';
  return JSON.stringify(pts);
}
"""


def pick(img: np.ndarray, title: str, mode: str = "polygon", crop=None, width: int = 1100) -> list:
    """Click points on `img` (full-res BGR). mode: polygon | polyline | line | box.
    crop=(x1,y1,x2,y2) zooms into a region (e.g. a traffic light). Returns full-res coordinates."""
    from IPython.display import Javascript, display
    from google.colab.output import eval_js
    x0, y0 = 0, 0
    view = img
    if crop is not None:
        x0, y0, x1, y1 = [int(v) for v in crop]
        view = img[y0:y1, x0:x1]
    s = width / view.shape[1]
    small = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    b64 = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    display(Javascript(_JS))
    pts = json.loads(eval_js(f"wiutPick('{b64}', {small.shape[1]}, {small.shape[0]}, {json.dumps(title)}, '{mode}')"))
    return [[round(x / s + x0, 1), round(y / s + y0, 1)] for x, y in pts]


def box_from(pts):
    (xa, ya), (xb, yb) = pts
    return [int(min(xa, xb)), int(min(ya, yb)), int(max(xa, xb)), int(max(ya, yb))]


def direction_from(pts):
    (xa, ya), (xb, yb) = pts
    v = np.array([xb - xa, yb - ya], float)
    return (v / max(np.linalg.norm(v), 1e-6)).round(4).tolist()


def play_clip(video: str, t0: float, t1: float, width: int = 960):
    """Inline HTML5 player for a short clip - check your label boundaries frame-accurately."""
    from IPython.display import HTML, display
    out = f"/tmp/clip_{os.path.basename(video)}_{t0:.1f}_{t1:.1f}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0, t0):.2f}", "-to", f"{t1:.2f}", "-i", video,
                    "-vf", f"scale={width}:-2,drawtext=text='%{{pts\\:flt\\:{max(0, t0):.2f}}}':x=10:y=10:fontsize=28:fontcolor=yellow:box=1:boxcolor=black",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an", out], check=True)
    b64 = base64.b64encode(open(out, "rb").read()).decode()
    display(HTML(f'<video controls width="{width}" src="data:video/mp4;base64,{b64}"></video>'))
