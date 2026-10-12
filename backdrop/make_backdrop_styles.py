"""S2S Fit event photo backdrops: cinematic style variations.

Usage: python3 make_backdrop_styles.py <logo.png> <out_dir> <size_px> <style|all>
styles: arena, inferno, smoke, gate, kinetic, concrete
"""
import sys, os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RED = np.array([255, 49, 49], np.float32)
WHITE = np.array([255, 255, 255], np.float32)
WARM = np.array([255, 236, 220], np.float32)
FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"
n = 1600  # resolution of the smooth light field (upscaled to N)


# ---------------- helpers ----------------
def grid():
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    return x, y


def vnoise(scale, seed, size=n):
    r = np.random.default_rng(seed).random((scale, scale)).astype(np.float32)
    return np.asarray(Image.fromarray(r, "F").resize((size, size), Image.BICUBIC))


def fbm(base, octaves, seed, size=n):
    out, amp, tot = np.zeros((size, size), np.float32), 1.0, 0.0
    for o in range(octaves):
        out += amp * vnoise(base * 2 ** o, seed + o, size)
        tot += amp
        amp *= 0.5
    return out / tot


def warp(f, dx, dy, k):
    """domain-warp field f by offsets dx, dy (0..1 noise) with strength k px."""
    yy, xx = np.mgrid[0:n, 0:n]
    sx = np.clip((xx + (dx - 0.5) * k).astype(int), 0, n - 1)
    sy = np.clip((yy + (dy - 0.5) * k).astype(int), 0, n - 1)
    return f[sy, sx]


def rays(x, y, cx, cy, count, seed, sharp=3.0):
    """angular god-ray mask emanating from (cx, cy)."""
    th = (np.arctan2(y - cy, x - cx) / (2 * np.pi) + 0.5)  # 0..1
    sig = np.random.default_rng(seed).random(count).astype(np.float32)
    sig = np.concatenate([sig, sig[:1]])
    idx = th * count
    i0 = np.floor(idx).astype(int) % count
    f = idx - np.floor(idx)
    f = f * f * (3 - 2 * f)
    v = sig[i0] * (1 - f) + sig[i0 + 1] * f
    return v ** sharp


def blur(a, r):
    m = float(a.max()) or 1.0  # GaussianBlur needs an 8-bit mode
    im = Image.fromarray((a / m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im, np.float32) * m / 255


def gauss2(x, y, cx, cy, sx, sy):
    return np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def vignette(img, x, y, cy=0.42, inner=0.30, amt=0.8):
    v = np.hypot(x - 0.5, (y - cy) * 1.1)
    return img * (1 - amt * np.clip((v - inner) / 0.45, 0, 1) ** 1.8)[..., None]


def to_canvas(img, N):
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((N, N), Image.BICUBIC).convert("RGBA")


def load_logo(path):
    logo = Image.open(path).convert("RGBA")
    la = np.array(logo)
    la[..., 3][la[..., :3].mean(-1) < 80] = 0  # strip baked-in dark shadow
    logo = Image.fromarray(la)
    return logo.crop(logo.getchannel("A").getbbox())


def crisp(img, scale):
    """re-sharpen alpha edges after upscaling a small raster logo."""
    if scale <= 1.2:
        return img
    a = np.asarray(img.getchannel("A"), np.float32) / 255
    a = Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(scale * 0.35))
    a = np.asarray(a, np.float32) / 255
    band = 0.5 / scale  # ~1px anti-aliased edge at output size
    a = np.clip((a - 0.5) / (2 * band) + 0.5, 0, 1)
    out = img.copy()
    out.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    return out


def place_logo(canvas, logo, N, width_frac, cy_frac, halo=0.22, glow_rgb=None, glow_amt=0.0):
    lw = int(N * width_frac)
    lh = int(logo.height * lw / logo.width)
    lg = crisp(logo.resize((lw, lh), Image.LANCZOS), lw / logo.width)
    lx, ly = (N - lw) // 2, int(N * cy_frac - lh / 2)
    a = Image.new("L", (N, N), 0)
    a.paste(lg.getchannel("A"), (lx, ly))
    if glow_rgb is not None and glow_amt > 0:  # wide coloured bloom
        g = a.filter(ImageFilter.GaussianBlur(N * 0.03)).point(lambda p: int(min(255, p * glow_amt)))
        canvas = Image.alpha_composite(canvas, Image.merge("RGBA", (*Image.new("RGB", (N, N), glow_rgb).split(), g)))
    if halo > 0:
        h = a.filter(ImageFilter.GaussianBlur(N * 0.012)).point(lambda p: int(p * halo))
        canvas = Image.alpha_composite(canvas, Image.merge("RGBA", (*Image.new("RGB", (N, N), (255, 255, 255)).split(), h)))
    canvas.alpha_composite(lg, (lx, ly))
    bar_y = ly + int(lh * (491 - 342) / (738 - 342))  # barbell axis in cropped logo
    return canvas, (lx, ly, lw, lh, bar_y)


def bar_lines(canvas, N, box, alpha=70, color=(255, 255, 255)):
    lx, ly, lw, lh, bar_y = box
    t = max(2, int(N * 0.0016))
    xs = np.arange(N, dtype=np.float32)
    dist = np.where(xs < lx, lx - xs, np.where(xs > lx + lw, xs - (lx + lw), 0)) / (N * 0.5)
    fade = (np.clip(1 - dist, 0, 1) ** 1.5 * alpha).astype(np.uint8)
    strip = np.zeros((N, N), np.uint8)
    strip[bar_y - t // 2: bar_y + t // 2 + 1, :] = fade[None, :]
    l = Image.merge("RGBA", (*Image.new("RGB", (N, N), color).split(), Image.fromarray(strip)))
    return Image.alpha_composite(canvas, l)


def streak(canvas, N, y, alpha=150, color=(255, 49, 49), thick=0.004, blur_r=0.006, x0=0.02, x1=0.98):
    s = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(s).ellipse([int(N * x0), y - int(N * thick), int(N * x1), y + int(N * thick)], fill=(*color, alpha))
    return Image.alpha_composite(canvas, s.filter(ImageFilter.GaussianBlur(N * blur_r)))


def particles(N, count, region, seed, color=(255, 255, 255), size=(0.0006, 0.0025), alpha=(40, 200), blur_r=0.0008, bias=None):
    """region(x, y) -> weight 0..1 for rejection sampling."""
    rng = np.random.default_rng(seed)
    M = N // 2
    lay = Image.new("RGBA", (M, M), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    placed = 0
    while placed < count:
        px, py = rng.random(), rng.random()
        if rng.random() > region(px, py):
            continue
        r = M * (size[0] + (size[1] - size[0]) * rng.random() ** 3)
        a = int(alpha[0] + (alpha[1] - alpha[0]) * rng.random() ** 2)
        d.ellipse([px * M - r, py * M - r, px * M + r, py * M + r], fill=(*color, a))
        placed += 1
    lay = lay.filter(ImageFilter.GaussianBlur(M * blur_r))
    return lay.resize((N, N), Image.BICUBIC)


def footer(canvas, N, alpha=120):
    txt = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(txt)
    font = ImageFont.truetype(FONT, int(N * 0.0125))
    label = "  ".join("S2SFIT.COM")
    w = d.textlength(label, font=font)
    ty = int(N * 0.905)
    d.text(((N - w) / 2, ty), label, font=font, fill=(255, 255, 255, alpha))
    t, dash, gap = max(2, int(N * 0.0016)), int(N * 0.03), N * 0.015
    gy = ty + int(N * 0.0125 * 0.62)
    d.rectangle([(N - w) / 2 - dash - gap, gy - t // 2, (N - w) / 2 - gap, gy + t // 2], fill=(255, 49, 49, 220))
    d.rectangle([(N + w) / 2 + gap, gy - t // 2, (N + w) / 2 + dash + gap, gy + t // 2], fill=(255, 49, 49, 220))
    return Image.alpha_composite(canvas, txt)


def finish(canvas, N, grain=3.4, seed=7):
    arr = np.asarray(canvas.convert("RGB"), np.int16)
    noise = np.random.default_rng(seed).normal(0, grain, (N, N, 1)).astype(np.int16)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


# ---------------- styles ----------------
def style_arena(logo, N):
    """Hard overhead spotlight, volumetric haze and dust, pool of light on the floor."""
    x, y = grid()
    img = np.full((n, n, 3), 4.0, np.float32)
    top = (0.5, -0.12)
    half = 0.03 + 0.36 * np.clip(y - top[1], 0, None)
    edge = np.clip((half - np.abs(x - 0.5)) / (0.05 + 0.04 * y), 0, 1)  # soft-edged cone
    haze = 0.55 + 0.45 * fbm(5, 5, 11)
    ry = 0.65 + 0.35 * rays(x, y, *top, 220, 3, 2.0)
    cone = edge * haze * ry * (0.55 + 0.45 * np.exp(-((y - 0.35) / 0.6) ** 2))
    img += 62 * cone[..., None] * (WARM / 255)
    img += 70 * gauss2(x, y, 0.5, 0.97, 0.36, 0.05)[..., None] * (WARM / 255)  # floor pool
    img += 26 * gauss2(x, y, 0.5, 0.36, 0.30, 0.20)[..., None]
    img += 0.07 * gauss2(x, y, 0.5, 0.50, 0.45, 0.035)[..., None] * RED
    img = vignette(img, x, y, amt=0.85)
    c = to_canvas(img, N)

    def reg(px, py):
        hw = 0.03 + 0.36 * (py + 0.12)
        return 0.9 if abs(px - 0.5) < hw * 0.9 else 0.0
    c = Image.alpha_composite(c, particles(N, 2200, reg, 5, alpha=(20, 140)))
    c, box = place_logo(c, logo, N, 0.48, 0.36, halo=0.25)
    c = streak(c, N, box[4], alpha=110)
    return footer(c, N)


def style_inferno(logo, N):
    """Black sky, red horizon fire with god rays fanning up behind the logo, rising embers."""
    x, y = grid()
    hz = 0.60
    img = np.full((n, n, 3), 3.0, np.float32)
    below = np.clip((y - hz) / 0.4, 0, 1)
    glow = np.exp(-((y - hz) / 0.10) ** 2) * np.exp(-((x - 0.5) / 0.75) ** 2)
    img += glow[..., None] * RED * 0.55
    img += (np.exp(-((y - hz) / 0.012) ** 2) * np.exp(-((x - 0.5) / 0.45) ** 2))[..., None] * np.array([255, 170, 140]) * 0.9
    r = rays(x, y, 0.5, hz, 160, 21, 3.5) * (y < hz)
    fall = np.exp(-np.hypot(x - 0.5, (y - hz) * 1.3) / 0.38)
    img += (r * fall)[..., None] * RED * 0.55
    img += (blur(r * fall, 6))[..., None] * np.array([255, 120, 100]) * 0.25
    smoke = fbm(4, 5, 31)
    img += (smoke ** 2 * np.exp(-((y - hz + 0.08) / 0.18) ** 2))[..., None] * RED * 0.18
    img *= (1 - 0.65 * below ** 0.8)[..., None]  # ground falls to black
    img = vignette(img, x, y, cy=0.5, amt=0.7)
    c = to_canvas(img, N)

    def reg(px, py):
        return float(np.exp(-((py - hz + 0.12) / 0.22) ** 2)) * (0.3 + 0.7 * math.exp(-((px - 0.5) / 0.45) ** 2))
    c = Image.alpha_composite(c, particles(N, 900, reg, 9, color=(255, 90, 60), size=(0.0005, 0.0018), alpha=(60, 230)))
    c, box = place_logo(c, logo, N, 0.50, 0.36, halo=0.18, glow_rgb=(255, 49, 49), glow_amt=0.55)
    c = streak(c, N, int(N * hz), alpha=255, color=(255, 210, 200), thick=0.0018, blur_r=0.003, x0=0.08, x1=0.92)
    c = streak(c, N, int(N * hz), alpha=200, thick=0.006, blur_r=0.012, x0=-0.1, x1=1.1)
    return footer(c, N)


def style_smoke(logo, N):
    """Turbulent backlit smoke, red from behind the logo, cold white from above."""
    x, y = grid()
    a, b = fbm(3, 4, 41), fbm(3, 4, 42)
    s = warp(fbm(4, 6, 43), a, b, 420)
    s2 = warp(fbm(6, 5, 44), b, a, 260)
    raw = 0.6 * s + 0.4 * s2
    raw = 0.6 * raw + 0.4 * raw[:, ::-1]  # loosely symmetric so it frames the logo
    dens = np.clip((raw - 0.36) * 2.8, 0, 1) ** 1.4
    back = gauss2(x, y, 0.5, 0.40, 0.55, 0.36)
    top = np.exp(-((y + 0.1) / 0.55) ** 2) * np.exp(-((x - 0.5) / 0.5) ** 2)
    img = np.full((n, n, 3), 3.0, np.float32)
    img += (dens * back)[..., None] * RED * 0.62
    img += (dens * top)[..., None] * np.array([200, 205, 215]) * 0.45
    img += back[..., None] * RED * 0.06
    img += (blur(dens * back, 18))[..., None] * RED * 0.15
    img = vignette(img, x, y, amt=0.85)
    c = to_canvas(img, N)
    c, box = place_logo(c, logo, N, 0.50, 0.38, halo=0.30)
    c = bar_lines(c, N, box, alpha=60)
    return footer(c, N)


def style_gate(logo, N):
    """Two blazing vertical light slits flanking the logo, glossy floor reflections, red rims."""
    x, y = grid()
    img = np.full((n, n, 3), 3.0, np.float32)
    floor = 0.80
    for sx in (0.18, 0.82):
        core = np.exp(-((x - sx) / 0.0035) ** 2) * (y < floor)
        bloom = np.exp(-((x - sx) / 0.03) ** 2) * (y < floor)
        wide = np.exp(-((x - sx) / 0.12) ** 2)
        img += core[..., None] * WHITE * 1.2
        img += bloom[..., None] * WARM * 0.35
        img += wide[..., None] * RED * 0.10
        refl = np.exp(-((x - sx) / (0.006 + 0.05 * np.clip(y - floor, 0, 1))) ** 2)
        refl = refl * (y >= floor) * np.exp(-(y - floor) / 0.12)
        img += refl[..., None] * WARM * 0.45
    # red rim on far edges
    img += (np.exp(-(x / 0.05) ** 2) + np.exp(-((1 - x) / 0.05) ** 2))[..., None] * RED * 0.22
    img += gauss2(x, y, 0.5, 0.38, 0.28, 0.22)[..., None] * 22
    fl = (y >= floor) * np.exp(-((x - 0.5) / 0.6) ** 2) * np.exp(-(y - floor) / 0.05)
    img += fl[..., None] * np.array([60, 20, 20])
    img += (np.exp(-((y - floor) / 0.0025) ** 2) * np.exp(-((x - 0.5) / 0.55) ** 2))[..., None] * RED * 0.6
    haze = fbm(5, 4, 51)
    img *= (0.85 + 0.3 * haze)[..., None]
    img = vignette(img, x, y, amt=0.6)
    c = to_canvas(img, N)
    c, box = place_logo(c, logo, N, 0.46, 0.38, halo=0.22)
    c = bar_lines(c, N, box, alpha=50)
    return footer(c, N)


def style_kinetic(logo, N):
    """Motion-blurred red/white light streaks slashing through a band behind the logo."""
    x, y = grid()
    rng = np.random.default_rng(61)
    M = n * 2
    lay = Image.new("RGB", (M, M), (0, 0, 0))
    d = ImageDraw.Draw(lay)
    for _ in range(260):
        cy = rng.normal(0.5, 0.05) * M
        cx = rng.random() * M
        L = M * (0.04 + 0.35 * rng.random() ** 2)
        w = max(1, int(M * 0.0005 * (0.5 + rng.random() * 2.5)))
        red = rng.random() < 0.75
        k = 0.15 + 0.85 * rng.random() ** 3
        col = tuple(int(c * k) for c in ((255, 49, 49) if red else (255, 240, 230)))
        d.line([cx - L / 2, cy, cx + L / 2, cy], fill=col, width=w)
    # horizontal motion smear: squash then stretch along x
    lay = lay.resize((M // 24, M), Image.BILINEAR).resize((M, M), Image.BICUBIC)
    lay = lay.rotate(-12, resample=Image.BICUBIC)
    m = M // 2
    lay = lay.crop((M // 2 - m // 2, M // 2 - m // 2, M // 2 + m // 2, M // 2 + m // 2)).resize((n, n), Image.BICUBIC)
    st = np.asarray(lay, np.float32)
    st = np.roll(st, int(n * (0.38 - 0.5)), axis=0)
    img = np.full((n, n, 3), 3.0, np.float32)
    img += st * 1.4
    img += blur(st.mean(-1), 14)[..., None] * RED / 255 * 1.6
    img += gauss2(x, y, 0.5, 0.38, 0.4, 0.2)[..., None] * 14
    img = vignette(img, x, y, amt=0.85)
    c = to_canvas(img, N)
    plate = Image.new("L", (N, N), 0)
    ImageDraw.Draw(plate).ellipse([int(N * 0.22), int(N * 0.27), int(N * 0.78), int(N * 0.49)], fill=150)
    plate = plate.filter(ImageFilter.GaussianBlur(N * 0.05))
    c = Image.alpha_composite(c, Image.merge("RGBA", (*Image.new("RGB", (N, N), (0, 0, 0)).split(), plate)))
    c, box = place_logo(c, logo, N, 0.48, 0.38, halo=0.28)
    return footer(c, N)


def style_concrete(logo, N):
    """Raw concrete wall, hard raking light from upper-left, logo as lit signage with red underglow."""
    x, y = grid()
    tex = 0.55 * fbm(6, 6, 71) + 0.30 * fbm(40, 3, 72) + 0.15 * vnoise(n, 73)
    pits = (vnoise(n // 3, 74) > 0.985) * 0.5
    tex = np.clip(tex - pits * 0.6, 0, 1)
    # board-form panel seams
    seams = np.zeros_like(x)
    for sy in (0.25, 0.5, 0.75):
        seams += np.exp(-((y - sy) / 0.0018) ** 2) * 0.35
    for sx in (0.333, 0.667):
        seams += np.exp(-((x - sx) / 0.0016) ** 2) * 0.35
    rake = np.exp(-np.hypot(x - 0.05, y + 0.05) / 0.55)
    light = 6 + 95 * rake + 22 * gauss2(x, y, 0.5, 0.38, 0.32, 0.2)
    lum = light * (0.55 + 0.75 * tex) * (1 - seams)
    img = lum[..., None] * (np.array([235, 230, 225]) / 255)
    img += gauss2(x, y, 0.5, 0.52, 0.40, 0.06)[..., None] * RED * 0.22 * (0.6 + 0.6 * tex[..., None])
    img = vignette(img, x, y, amt=0.88)
    c = to_canvas(img, N)
    c, box = place_logo(c, logo, N, 0.46, 0.38, halo=0.30, glow_rgb=(255, 49, 49), glow_amt=0.35)
    c = streak(c, N, box[4], alpha=90, x0=0.1, x1=0.9)
    return footer(c, N)


STYLES = {
    "arena": style_arena, "inferno": style_inferno, "smoke": style_smoke,
    "gate": style_gate, "kinetic": style_kinetic, "concrete": style_concrete,
}

if __name__ == "__main__":
    logo_path, out_dir, N, which = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)
    logo = load_logo(logo_path)
    for i, name in enumerate(STYLES if which == "all" else [which]):
        out = finish(STYLES[name](logo, N), N, seed=100 + i)
        stem = os.path.join(out_dir, f"S2S_Backdrop_{name}_8x8ft")
        out.save(stem + ".jpg", quality=95, dpi=(N / 96, N / 96), subsampling=0)
        out.resize((1600, 1600), Image.LANCZOS).save(stem + "_preview.jpg", quality=90)
        print("saved", stem)
