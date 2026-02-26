import numpy as np
import soundfile as sf

def crossfade_concat(a, b, sr=24000, fade_ms=30):
    fade = int(sr * fade_ms / 1000)

    # Trường hợp audio quá ngắn
    if len(a) < fade or len(b) < fade:
        return np.concatenate([a, b])

    a_end = a[-fade:]
    b_start = b[:fade]

    fade_out = np.linspace(1.0, 0.0, fade)
    fade_in  = np.linspace(0.0, 1.0, fade)

    mixed = a_end * fade_out + b_start * fade_in

    return np.concatenate([
        a[:-fade],
        mixed,
        b[fade:]
    ])


# Load audio
a, sr = sf.read("check4_vi.wav")
b, sr2 = sf.read("check5_en.wav")

assert sr == sr2, "Sample rate mismatch!"

# 1. Nối thẳng (baseline – để nghe lỗi)
plain_concat = np.concatenate([a, b])
sf.write("plain_concat.wav", plain_concat, sr)

# 2. Nối có crossfade
crossfaded = crossfade_concat(a, b, sr=sr, fade_ms=30)
sf.write("crossfade_concat.wav", crossfaded, sr)

print("Saved:")
print(" - plain_concat.wav  (nghe sẽ thấy 'cục')")
print(" - crossfade_concat.wav (mượt hơn rõ)")
