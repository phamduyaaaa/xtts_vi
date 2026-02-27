import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Đang load model để trích xuất giọng...")
config = XttsConfig()
config.load_json("model/config.json")
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path="model/model.pth", vocab_path="model/vocab.json", use_deepspeed=False)

# Trích xuất từ file WAV
speaker_audio_file = "model/vi_man.wav"
print(f"Đang phân tích file: {speaker_audio_file}")
gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

# LƯU RA FILE .pth
output_file = "model/vi_man_latents.pth"
torch.save({
    "gpt_cond_latent": gpt_cond_latent,
    "speaker_embedding": speaker_embedding
}, output_file)

print(f"✅ Đã lưu thành công Vector giọng nói ra file: {output_file}")
