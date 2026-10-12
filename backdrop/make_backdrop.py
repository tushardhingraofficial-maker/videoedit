"""S2S Fit event photo backdrop generator.

Usage: python3 make_backdrop.py <logo.png> <out_dir> <size_px> [variant]
variant: hero (default) | wall
"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logo_path, out_dir, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
variant = sys.argv[4] if len(sys.argv) > 4 else "hero"
RED = np.array([255, 49, 49], np.float32)
FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"

# ---------- low-res light field (smooth, upscaled later) ----------
n = 1200
y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
LOGO_CY = 0.34  # logo sits at head height in photos

img = np.full((n, n, 3), 6.0, np.float32)  # near-black base

# soft key light behind the logo
r = np.hypot((x - 0.5) / 1.0, (y - LOGO_CY) / 0.8)
img += 30 * np.exp(-(r / 0.36) ** 2)[..., None]

# overhead volumetric cone
half_w = 0.06 + 0.42 * np.clip(y, 0, 1)
cone = np.clip(1 - np.abs(x - 0.5) / half_w, 0, 1) ** 1.6
cone *= np.exp(-((y - 0.05) / 0.75) ** 2)
img += 16 * cone[..., None]

# red horizon bleed under the logo
rb = np.exp(-(((x - 0.5) / 0.55) ** 2 + ((y - (LOGO_CY + 0.13)) / 0.05) ** 2))
img += rb[..., None] * RED * 0.10

# floor falloff + vignette
img *= (1 - 0.55 * np.clip((y - 0.62) / 0.38, 0, 1) ** 1.3)[..., None]
v = np.hypot(x - 0.5, (y - 0.42) * 1.1)
img *= (1 - 0.75 * np.clip((v - 0.30) / 0.45, 0, 1) ** 1.8)[..., None]

base = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((N, N), Image.BICUBIC)
canvas = base.convert("RGBA")

# ---------- logo ----------
logo = Image.open(logo_path).convert("RGBA")
# strip the faint dark drop-shadow baked into the "transparent" PNG
la = np.array(logo)
la[..., 3][la[..., :3].mean(-1) < 80] = 0
logo = Image.fromarray(la)
logo = logo.crop(logo.getchannel("A").getbbox())
lw = int(N * (0.50 if variant == "hero" else 0.36))
lh = int(logo.height * lw / logo.width)
logo = logo.resize((lw, lh), Image.LANCZOS)
lx, ly = (N - lw) // 2, int(N * LOGO_CY - lh / 2)
# barbell axis sits at ~ (491-342)/(738-342) of the cropped logo height
bar_y = ly + int(lh * (491 - 342) / (738 - 342))

def layer():
    return Image.new("RGBA", (N, N), (0, 0, 0, 0))

# barbell extension lines: logo's bar continues to the edges, fading out
lines = layer()
d = ImageDraw.Draw(lines)
t = max(2, int(N * 0.0016))
d.rectangle([0, bar_y - t // 2, N, bar_y + t // 2], fill=(255, 255, 255, 255))
xs = np.arange(N, dtype=np.float32)
dist = np.where(xs < lx, lx - xs, np.where(xs > lx + lw, xs - (lx + lw), 0)) / (N * 0.5)
fade = (np.clip(1 - dist, 0, 1) ** 1.5 * 70).astype(np.uint8)
a = np.array(lines.getchannel("A"), np.uint16) * fade[None, :] // 255
lines.putalpha(Image.fromarray(a.astype(np.uint8)))
canvas = Image.alpha_composite(canvas, lines)

# anamorphic red streak along the bar axis
streak = layer()
ImageDraw.Draw(streak).ellipse(
    [int(N * 0.02), bar_y - int(N * 0.004), int(N * 0.98), bar_y + int(N * 0.004)],
    fill=(255, 49, 49, 150))
streak = streak.filter(ImageFilter.GaussianBlur(N * 0.006))
canvas = Image.alpha_composite(canvas, streak)

if variant == "hero":
    gw = int(N * 1.25)
    ghost = logo.resize((gw, int(logo.height * gw / logo.width)), Image.LANCZOS)
    ghost = Image.merge("RGBA", (*Image.new("RGB", ghost.size, (255, 255, 255)).split(),
                                 ghost.getchannel("A").point(lambda p: int(p * 0.035))))
    g = layer()
    g.alpha_composite(ghost, ((N - gw) // 2, int(N * 0.60)))
    # fade toward the floor
    fm = Image.linear_gradient("L").resize((N, N)).point(lambda p: 255 - p)
    g.putalpha(Image.fromarray((np.array(g.getchannel("A"), np.uint16) * np.array(fm) // 255).astype(np.uint8)))
    canvas = Image.alpha_composite(canvas, g)

# halo behind logo for separation
halo_a = Image.new("L", (N, N), 0)
halo_a.paste(logo.getchannel("A"), (lx, ly))
halo_a = halo_a.filter(ImageFilter.GaussianBlur(N * 0.014)).point(lambda p: int(p * 0.20))
halo = Image.new("RGBA", (N, N), (255, 255, 255, 0))
halo.putalpha(halo_a)
canvas = Image.alpha_composite(canvas, halo)
canvas.alpha_composite(logo, (lx, ly))

# ---------- step-and-repeat (wall variant) ----------
if variant == "wall":
    small_w = int(N * 0.13)
    sm = logo.resize((small_w, int(logo.height * small_w / logo.width)), Image.LANCZOS)
    sm.putalpha(sm.getchannel("A").point(lambda p: int(p * 0.30)))
    cols, rows = 4, 5
    cw, rh = N / cols, N / rows
    keep_out = (lx - N * 0.04, ly - N * 0.05, lx + lw + N * 0.04, ly + lh + N * 0.05)
    for rI in range(rows):
        for cI in range(cols + 1):
            cx = (cI + (0.5 if rI % 2 == 0 else 0.0)) * cw
            cy = (rI + 0.5) * rh
            bx, by = int(cx - sm.width / 2), int(cy - sm.height / 2)
            if bx + sm.width < 0 or bx > N:
                continue
            if not (bx + sm.width < keep_out[0] or bx > keep_out[2]
                    or by + sm.height < keep_out[1] or by > keep_out[3]):
                continue
            if by + sm.height > N * 0.88:  # footer zone
                continue
            if abs(cy - bar_y) < sm.height:  # don't sit on the bar line
                continue
            canvas.alpha_composite(sm, (bx, by))

# ---------- footer ----------
txt = layer()
d = ImageDraw.Draw(txt)
font = ImageFont.truetype(FONT, int(N * 0.0125))
label = "  ".join("S2SFIT.COM")
w = d.textlength(label, font=font)
ty = int(N * 0.905)
d.text(((N - w) / 2, ty), label, font=font, fill=(255, 255, 255, 120))
dash = int(N * 0.03)
gy = ty + int(N * 0.0125 * 0.62)
d.rectangle([(N - w) / 2 - dash - N * 0.015, gy - t // 2, (N - w) / 2 - N * 0.015, gy + t // 2], fill=(255, 49, 49, 220))
d.rectangle([(N + w) / 2 + N * 0.015, gy - t // 2, (N + w) / 2 + dash + N * 0.015, gy + t // 2], fill=(255, 49, 49, 220))
canvas = Image.alpha_composite(canvas, txt)

# ---------- film grain (also dithers gradient banding) ----------
arr = np.asarray(canvas.convert("RGB"), np.int16)
rng = np.random.default_rng(7)
noise = rng.normal(0, 3.2, (N, N, 1)).astype(np.int16)
arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
out = Image.fromarray(arr)

os.makedirs(out_dir, exist_ok=True)
stem = os.path.join(out_dir, f"S2S_Backdrop_{variant}_8x8ft")
out.save(stem + ".jpg", quality=95, dpi=(N / 96, N / 96), subsampling=0)
out.resize((1600, 1600), Image.LANCZOS).save(stem + "_preview.jpg", quality=90)
print("saved", stem)
