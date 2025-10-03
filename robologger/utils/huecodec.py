# This file is adapted from https://github.com/cheind/hue-depth-encoding v1.1.0

# MIT License

# Copyright (c) 2024 Christoph Heindl

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from contextlib import contextmanager
from typing import Optional, Tuple

# import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def rgb2hsv(
    rgb: np.ndarray, *, output: np.ndarray = None, ftype: np.dtype = np.float32
):
    """Vectorized RGB to HSV"""

    if output is None:
        output = np.empty_like(rgb)
    output[:] = 0
    h, s, v = np.split(output, 3, -1)
    h = h.squeeze(-1)
    s = s.squeeze(-1)
    v = v.squeeze(-1)

    rgb_amax = rgb.argmax(-1)
    rgb_max = rgb.max(-1)
    rgb_min = rgb.min(-1)

    r = (rgb_max - rgb_min).astype(ftype)
    ok = r > 0

    # fmt: off
    m = ok & (rgb_amax == 0); h[m] = 0+(rgb[m, 1] - rgb[m, 2]) / r[m]
    m = ok & (rgb_amax == 1); h[m] = 2+(rgb[m, 2] - rgb[m, 0]) / r[m]
    m = ok & (rgb_amax == 2); h[m] = 4+(rgb[m, 0] - rgb[m, 1]) / r[m]
    # fmt: on
    h[:] *= 60
    h[h < 0] += 360

    s[ok] = r[ok] / rgb_max[ok]
    v[:] = rgb_max

    return np.stack((h, s, v), -1)


def hsv2rgb(hsv: np.ndarray, *, output: np.ndarray = None):

    h, s, v = np.split(hsv, 3, axis=-1)
    h = h.squeeze(-1)
    s = s.squeeze(-1)
    v = v.squeeze(-1)

    h = h / 60
    hi = np.floor(h).astype(int)
    f = h - hi
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    w = hi % 6
    if output is None:
        output = np.empty_like(hsv)
    # fmt: off
    m = w==0; output[m,0],output[m,1],output[m,2] = v[m],t[m],p[m]
    m = w==1; output[m,0],output[m,1],output[m,2] = q[m],v[m],p[m]
    m = w==2; output[m,0],output[m,1],output[m,2] = p[m],v[m],t[m]
    m = w==3; output[m,0],output[m,1],output[m,2] = p[m],q[m],v[m]
    m = w==4; output[m,0],output[m,1],output[m,2] = t[m],p[m],v[m]
    m = w==5; output[m,0],output[m,1],output[m,2] = v[m],p[m],q[m]
    # fmt: on

    return output


class EncoderOpts:
    def __init__(
        self,
        max_hue: float = 300,
        qtype: np.dtype = np.uint8,
        ftype: np.dtype = np.float32,
        err_depth: float = np.nan,
        err_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        min_v: float = 0.4,
        min_s: float = 0.4,
        use_lut: bool = True,
    ):
        assert np.issubdtype(qtype, np.unsignedinteger)
        self.qinfo = np.iinfo(qtype)
        self.qtype = qtype
        self.ftype = ftype
        self.max_hue = max_hue
        self.num_unique = int(self.max_hue / 60) * (2**self.qinfo.bits - 1) + 1
        self.bits = math.log(self.num_unique) / math.log(2)
        self.err_depth = err_depth
        self.err_rgb = np.array(err_rgb).astype(ftype)
        self.err_hsv = rgb2hsv(self.err_rgb[None]).squeeze().astype(ftype)
        self.min_v = min_v
        self.min_s = min_s
        self.use_lut = use_lut

        self._enc_lut = None
        self._dec_lut = None

    @property
    def enc_lut(self):
        if self.use_lut:
            if self._enc_lut is None:
                self._enc_lut = _create_enc_lut(self)
        return self._enc_lut

    @property
    def dec_lut(self):
        if self.use_lut:
            if self._dec_lut is None:
                self._dec_lut = _create_dec_lut(self)
        return self._dec_lut


# Default encoder settings
_default_opts = EncoderOpts(use_lut=True)


@contextmanager
def enc_opts(new_opts: EncoderOpts = EncoderOpts()):
    global _default_opts
    try:
        old_opts = _default_opts
        _default_opts = new_opts
        yield new_opts
    finally:
        _default_opts = old_opts


def encode(
    depth: np.ndarray,
    *,
    output: np.ndarray = None,
    sanitized: bool = False,
    opts: EncoderOpts = None,
):
    opts = opts or _default_opts
    h = depth * opts.max_hue
    s = np.ones_like(h)
    v = s
    if not sanitized:
        ok = np.isfinite(depth) & (depth >= 0) & (depth <= 1.0)
        h[~ok] = opts.err_hsv[0]
        s[~ok] = opts.err_hsv[1]
        v[~ok] = opts.err_hsv[2]

    hsv = np.stack((h, s, v), -1)
    rgb = hsv2rgb(hsv, output=output)
    return rgb


def decode(
    rgb: np.ndarray,
    *,
    output: np.ndarray = None,
    opts: EncoderOpts = None,
):
    opts = opts or _default_opts
    if np.issubdtype(rgb.dtype, np.unsignedinteger):
        rgb = dequantize(rgb)
    hsv = rgb2hsv(rgb)

    if output is None:
        output = np.empty(rgb.shape[:-1], dtype=opts.ftype)

    ok = (
        (hsv[..., 1] >= opts.min_s)
        & (hsv[..., 2] >= opts.min_v)
        & (hsv[..., 0] <= opts.max_hue)
    )
    output[:] = opts.err_depth
    output[ok] = hsv[ok, 0] / opts.max_hue

    return output


def quantize(x: np.ndarray, *, opts: EncoderOpts = None):
    opts = opts or _default_opts
    return np.round(np.iinfo(opts.qtype).max * x).astype(opts.qtype)


def dequantize(x: np.ndarray, *, opts: EncoderOpts = None):
    opts = opts or _default_opts
    return (x / np.iinfo(x.dtype).max).astype(opts.ftype)


def decode_lut(
    rgb: np.ndarray,
    *,
    output: np.ndarray = None,
    opts: EncoderOpts = _default_opts,
):
    opts = opts or None
    if output is None:
        output = np.empty(rgb.shape[:-1], dtype=opts.dec_lut.dtype)

    output[:] = opts.dec_lut[
        rgb[..., 0],
        rgb[..., 1],
        rgb[..., 2],
    ]
    return output


def encode_lut(
    depth: np.ndarray,
    *,
    output: np.ndarray = None,
    sanitized: bool = True,
    opts: EncoderOpts = None,
):
    opts = opts or _default_opts
    if output is None:
        output = np.empty(depth.shape + (3,), dtype=opts.qtype)

    # NOTE! this already reports quantized rgb, see _create_enc_lut
    # linspace inverse for index

    idx = np.round(depth * (opts.num_unique - 1)).astype(int)

    if sanitized:
        output[:] = opts.enc_lut[idx]
    else:
        ok = (idx >= 0) & (idx < opts.num_unique)
        output[:] = opts.err_rgb
        output[ok] = opts.enc_lut[idx[ok]]

    return output


def _create_enc_lut(opts: EncoderOpts = None):
    opts = opts or _default_opts
    d = np.linspace(0, 1.0, opts.num_unique, dtype=opts.ftype)
    return quantize(encode(d, opts=opts), opts=opts)


def _create_dec_lut(opts: EncoderOpts = None):
    opts = opts or _default_opts
    rgb = np.stack(
        np.meshgrid(
            np.arange(opts.qinfo.max + 1, dtype=opts.qtype),
            np.arange(opts.qinfo.max + 1, dtype=opts.qtype),
            np.arange(opts.qinfo.max + 1, dtype=opts.qtype),
            indexing="ij",
        ),
        -1,
    )

    return decode(rgb, opts=opts)


def depth2rgb(
    d: np.ndarray,
    zrange: Tuple[float, float],
    *,
    output: np.ndarray = None,
    sanitized: bool = False,
    inv_depth: bool = False,
    opts: EncoderOpts = None,
):
    """Compress depth to RGB

    The colorization process requires fitting a 16-bit depth map into a 10.5-bit color image.
    We limit the depth range to a subset of the 0-65535 range and re-normalize before colorization.

    With disparity encoding we actually encode 1/depth with the property that for closer depths
    the quantization is finer and coarser for larger depth values. Note that NaN and inf values
    are mapped to zrange min.

    Params:
        d: (*,) depth map
        zrange: clipping range for depth values before normalization to [0..HUE_ENCODER_MAX]
        inv_depth: colorizes 1/depth with finer quantization for closer depths
        opts: encoder options

    Returns:
        rgb: quantized color encoded depth map
    """
    opts = opts or _default_opts
    if inv_depth:
        zmin = 1 / zrange[1]
        zmax = 1 / zrange[0]
        zrange = (zmin, zmax)
        with np.errstate(divide="ignore"):
            d = 1 / d

    d = (d - zrange[0]) / (zrange[1] - zrange[0])
    return (
        encode_lut(d, output=output, sanitized=sanitized, opts=opts)
        if opts.use_lut
        else encode(d, output=output, sanitized=sanitized, opts=opts)
    )


def rgb2depth(
    rgb: np.ndarray,
    zrange: Tuple[float, float],
    *,
    output: np.ndarray = None,
    inv_depth: bool = False,
    opts: EncoderOpts = None,
):
    """Decompress RGB to depth

    See `depth2rgb` for explanation of parameters.

    Params:
        rgb: (*,3) color map
        zrange: zrange used during compression
        inv_depth: wether depth or disparity was encoded
        opts: encoder options

    Returns:
        rgb: color encoded depth map
    """
    opts = opts or _default_opts
    d = (
        decode_lut(rgb, output=output, opts=opts)
        if opts.use_lut
        else decode(rgb, output=output, opts=opts)
    )
    if inv_depth:
        zmin = 1 / zrange[1]
        zmax = 1 / zrange[0]
        zrange = (zmin, zmax)

    d = d * (zrange[1] - zrange[0]) + zrange[0]

    if inv_depth:
        with np.errstate(divide="ignore"):
            d = 1 / d
    return d


# def main():
#     h = np.arange(330)
#     s = np.ones(330)
#     v = np.ones(330)
#     hsv = np.stack((h, s, v), -1)
#     rgb = hsv2rgb(hsv)
#     hsv2 = rgb2hsv(rgb)
#     print(abs(hsv - hsv2).sum(-1).max())

#     # fig, ax = plt.subplots()
#     # ax.imshow(rgb[None], aspect="auto")
#     # plt.show()

#     # d = np.linspace(0, 1.0, 1276)
#     with enc_opts() as opts:
#         print(opts.max_hue, opts.num_unique)
#         d = np.linspace(0, 1.0, opts.num_unique)
#         rgb_enc = encode(d)

#         print((rgb_enc * 255)[:10])
#         rgb_byte = quantize(rgb_enc)
#         num_diff = (abs(np.diff(rgb_byte, axis=0)).sum(-1) > 0).sum() + 1
#         print(num_diff)

#         fig, ax = plt.subplots()
#         ax.imshow(rgb[None], aspect="auto", extent=(0, 1, 0, 1))

#         fig, ax = plt.subplots()
#         dd = decode(rgb_byte / 255)
#         print(rgb_byte)
#         print("here", abs(d - dd).max())

#         plt.plot(d, dd)
#         plt.show()

#         rgb_byte_lut = encode_lut(d)
#         assert (rgb_byte.astype(int) - rgb_byte_lut.astype(int)).sum() == 0

#         dd_lut = decode_lut(rgb_byte_lut)
#         print(abs(dd - dd_lut).max())

#         print(rgb_byte[:3])
#         print(rgb_byte_lut[:3])

#         rng = np.random.default_rng(123)
#         d = rng.random((512, 512), dtype=np.float32)
#         rgb = encode(d)
#         qrgb = quantize(rgb)
#         dr = decode(qrgb)
#         err_abs = abs(dr - d).max()
#         print(err_abs)

#         # print(_default.max_hue, _default.num_unique, _default.bits)

#         d = np.logspace(-5, 0, 100)
#         rgb = encode(d)
#         qrgb = quantize(rgb)
#         dr = decode(qrgb)
#         err_abs = abs(dr - d)
#         print(err_abs.max(), np.argmax(err_abs))

#         # print(np.argmax())

#         rgb = np.array(
#             [[10, 20, 50], [255, 0, 41], [201, 114, 255], [111, 63, 140]],
#             dtype=np.uint8,
#         )
#         # first two should be nan (value/saturation) and above ENC_MAX_HUE
#         # last one and second last should be 277.1/300 ~ 0.92 (desaturated ~55%, devalued ~55%)

#         d = decode(rgb)
#         print(d)

#     pass


# if __name__ == "__main__":
#     main()

def depth2logrgb(depth: np.ndarray, zrange: Tuple[float, float], opts: Optional[EncoderOpts] = None) -> npt.NDArray[np.uint8]:
    opts = opts or _default_opts
    depth_clipped = np.clip(depth, zrange[0], zrange[1])
    depth_logged = np.log(1 + depth_clipped) / np.log(1 + zrange[1])
    return depth2rgb(depth_logged, zrange, opts=opts)

def logrgb2depth(rgb: np.ndarray, zrange: Tuple[float, float], opts: Optional[EncoderOpts] = None) -> npt.NDArray[np.float32]:
    opts = opts or _default_opts
    depth_logged = rgb2depth(rgb, zrange, opts=opts)
    depth = np.exp(depth_logged * np.log(1 + zrange[1])) - 1
    return depth