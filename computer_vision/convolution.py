import numpy as np

def pad2d(x, p):
    if p == 0:
        return x
    h, w = x.shape[-2:]
    print(h, w)
    out = np.zeros(x.shape[:-2] + (h + 2 * p, w + 2 * p), dtype=x.dtype)
    out[..., p:p + h, p:p + w] = x
    return out

def pad2d_test():
    x = np.arange(9).reshape(3, 3)
    out = pad2d(x, 1)
    assert out.shape == (5, 5), f"shape mismatch: {out.shape}"
    assert np.all(out[1:4, 1:4] == x), "inner content mismatch"
    assert np.all(out[0, :] == 0) and np.all(out[-1, :] == 0), "top/bottom padding not zero"
    assert np.all(out[:, 0] == 0) and np.all(out[:, -1] == 0), "left/right padding not zero"

    # p=0 應原樣返回
    out0 = pad2d(x, 0)
    assert out0 is x, "p=0 should return the original array"

    # batch input (N, C, H, W)
    xb = np.ones((2, 3, 4, 4))
    outb = pad2d(xb, 2)
    assert outb.shape == (2, 3, 8, 8), f"batch shape mismatch: {outb.shape}"
    assert np.all(outb[:, :, 2:6, 2:6] == 1), "batch inner content mismatch"
    assert np.all(outb[:, :, :2, :] == 0), "batch top padding not zero"

    print("pad2d_test passed")

def conv2d_naive(x, w, b=None, stride=1, padding=0):
    c_in, h_in, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w

    x_pad = pad2d(x, padding)
    h_out = (h_in + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for oc in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                hs = i * stride
                ws = j * stride
                patch = x_pad[:, hs:hs + kh, ws:ws + kw]
                out[oc, i, j] = np.sum(patch * w[oc])
        if b is not None:
            out[oc] += b[oc]
    return out

def conv2d_naive_test():
    try:
        import torch
        import torch.nn.functional as F
        _torch = True
    except ImportError:
        _torch = False

    np.random.seed(0)

    configs = [
        dict(c_in=1, c_out=3, h=5, w=5, kh=3, kw=3, stride=1, padding=0),
        dict(c_in=2, c_out=4, h=6, w=6, kh=3, kw=3, stride=2, padding=1),
        dict(c_in=3, c_out=2, h=7, w=7, kh=5, kw=5, stride=1, padding=2),
    ]

    for cfg in configs:
        x = np.random.randn(cfg["c_in"], cfg["h"], cfg["w"]).astype(np.float32)
        w = np.random.randn(cfg["c_out"], cfg["c_in"], cfg["kh"], cfg["kw"]).astype(np.float32)
        b = np.random.randn(cfg["c_out"]).astype(np.float32)

        out = conv2d_naive(x, w, b, stride=cfg["stride"], padding=cfg["padding"])

        h_out = (cfg["h"] + 2 * cfg["padding"] - cfg["kh"]) // cfg["stride"] + 1
        w_out = (cfg["w"] + 2 * cfg["padding"] - cfg["kw"]) // cfg["stride"] + 1
        assert out.shape == (cfg["c_out"], h_out, w_out), \
            f"shape mismatch: {out.shape} vs ({cfg['c_out']}, {h_out}, {w_out})"

        if _torch:
            xt = torch.from_numpy(x).unsqueeze(0)
            wt = torch.from_numpy(w)
            bt = torch.from_numpy(b)
            ref = F.conv2d(xt, wt, bt, stride=cfg["stride"], padding=cfg["padding"])
            ref = ref.squeeze(0).numpy()
            assert np.allclose(out, ref, atol=1e-5), \
                f"value mismatch (cfg={cfg})\nours:\n{out}\ntorch:\n{ref}"

    print(f"conv2d_naive_test passed ({'with torch validation' if _torch else 'shape-only, torch not available'})")

def im2col(x, kh, kw, stride=1, padding=0):
    c_in, h, w = x.shape
    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    cols = np.zeros((c_in * kh * kw, h_out * w_out), dtype=x.dtype)
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            patch = x_pad[:, hs:hs + kh, ws:ws + kw]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, h_out, w_out

def im2col_test():
    # 用 conv2d_naive 的結果來驗證 im2col 的正確性
    np.random.seed(42)
    x = np.random.randn(2, 5, 5).astype(np.float32)   # (C, H, W)
    w = np.random.randn(4, 2, 3, 3).astype(np.float32) # (C_out, C_in, kH, kW)

    stride, padding = 1, 1

    # Ground truth via naive conv
    expected = conv2d_naive(x, w, stride=stride, padding=padding)

    # im2col + matmul
    cols, h_out, w_out = im2col(x, kh=3, kw=3, stride=stride, padding=padding)
    # w reshaped: (C_out, C_in*kH*kW)
    w_flat = w.reshape(4, -1)
    result = (w_flat @ cols).reshape(4, h_out, w_out)

    assert result.shape == expected.shape, f"shape mismatch: {result.shape} vs {expected.shape}"
    assert np.allclose(result, expected, atol=1e-5), \
        f"value mismatch:\nresult:\n{result}\nexpected:\n{expected}"
    print("im2col_test passed: shape =", result.shape)

    # 測試 no padding / stride=2
    x2 = np.random.randn(1, 6, 6).astype(np.float32)
    w2 = np.random.randn(3, 1, 3, 3).astype(np.float32)
    expected2 = conv2d_naive(x2, w2, stride=2, padding=0)
    cols2, h_out2, w_out2 = im2col(x2, kh=3, kw=3, stride=2, padding=0)
    result2 = (w2.reshape(3, -1) @ cols2).reshape(3, h_out2, w_out2)
    assert np.allclose(result2, expected2, atol=1e-5), "stride=2 test failed"
    print("im2col_test stride=2 passed: shape =", result2.shape)

def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    c_out, c_in, kh, kw = w.shape
    cols, h_out, w_out = im2col(x, kh, kw, stride, padding)
    w_flat = w.reshape(c_out, -1)
    out = w_flat @ cols
    if b is not None:
        out += b[:, None]
    return out.reshape(c_out, h_out, w_out)

if __name__ == "__main__":
    # pad2d_test()    
    # conv2d_naive_test()
    im2col_test()
