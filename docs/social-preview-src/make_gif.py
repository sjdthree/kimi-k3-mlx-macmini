"""Build docs/social-preview.gif (and .png) from frames.html.

Render the four frames with headless Chrome, then assemble them (needs Pillow):

    C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    for i in 0 1 2 3; do "$C" --headless=new --hide-scrollbars --force-device-scale-factor=1 \
      --window-size=1280,640 --screenshot=f$i.png "file://$PWD/frames.html#$i"; done
    python make_gif.py . && cp social-preview.gif ../ && cp f0.png ../social-preview.png

Frame 0 must work on its own: link previews (X, LinkedIn, Slack) show only the first frame.
Figures are from demos/private-document-analysis/local-results/summary.json.
"""
import sys
from pathlib import Path
from PIL import Image
d = Path(sys.argv[1])
order = [0, 1, 2, 3]
hold = {0: 4500, 1: 3800, 2: 3800, 3: 3800}
frames = [Image.open(d / f"f{i}.png").convert("RGB") for i in order]
# One shared adaptive palette keeps colors identical across frames (no flicker).
strip = Image.new("RGB", (1280, 640 * len(frames)))
for k, f in enumerate(frames):
    strip.paste(f, (0, 640 * k))
pal = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
out = d / "social-preview.gif"
q[0].save(out, save_all=True, append_images=q[1:], duration=[hold[i] for i in order],
          loop=0, optimize=True, disposal=1)
print(out, out.stat().st_size // 1024, "KiB")
