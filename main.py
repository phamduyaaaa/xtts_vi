import os
import time
import torch
import torchaudio
from vinorm import TTSnorm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH (OFFLINE)
# ==========================================
print("--- [1] ĐANG KHỞI TẠO HỆ THỐNG ---")
length_penalty = 1.0
repetition_penalty = 10.0
top_k = 10
top_p = 0.5
speed=0.9
temperature=0.7
num_beams=1
MODEL_DIR = "model/"
config_file = f"{MODEL_DIR}config.json"
model_weights = f"{MODEL_DIR}model.pth"
vocab_file = f"{MODEL_DIR}vocab.json"
speaker_audio_file = f"{MODEL_DIR}vi_man.wav"

# Kiểm tra thiết bị (GPU được ưu tiên để xử lý nhanh hơn)/home/ducduy/Downloads/final.wav
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[*] Chạy trên: {device.upper()}")

# Load cấu hình và trọng số
config = XttsConfig()
config.load_json(config_file)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config,
                            checkpoint_path=model_weights,
                            vocab_path=vocab_file,
                            use_deepspeed=False)
XTTS_MODEL.to(device)

# Trích xuất đặc trưng giọng mẫu
print("[*] Đang trích xuất đặc trưng giọng mẫu...")
gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)
print("--- KHỞI TẠO XONG! ---\n")

# ==========================================
# 2. HÀM TỔNG HỢP (KHÔNG CHUNKING)
# ==========================================
def generate_full_audio(model, text, language="vi"):
    # Chuẩn hóa văn bản (Xử lý số, ngày tháng, ký tự đặc biệt)
    if language == "vi":
        text = TTSnorm(text, unknown=False, lower=False, rule=True)
    
    print(f"⏳ Đang xử lý toàn bộ văn bản (Độ dài: {len(text)} ký tự)...")
    start_time = time.time()
    
    # Sử dụng torch.inference_mode để tối ưu tốc độ và bộ nhớ
    with torch.inference_mode():
        # Gọi trực tiếp model.inference với toàn bộ text, không chia nhỏ
        outputs = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            speed=speed,
            temperature=temperature,
            num_beams=num_beams,
        )
    
    
    end_time = time.time()
    print(f"✅ Xử lý xong trong: {end_time - start_time:.2f} giây.")
    
    # Chuyển đổi sang Tensor để lưu file
    audio_tensor = torch.tensor(outputs["wav"]).unsqueeze(0)
    return audio_tensor

# ==========================================
# 3. THỰC THI VÀ LƯU FILE
# ==========================================
if __name__ == "__main__":
    input_text = (
        '''
 Every day, try to spend at least 15 minutes practicing. Sự kiên trì chính là chìa khóa dẫn đến thành công. Consistency is the key to success. Đừng sợ mắc lỗi, vì đó là cách duy nhất để chúng ta tiến bộ hơn. 
        '''
    )
    
    output_filename = f"songngu_final_{length_penalty}_{repetition_penalty}_{top_k}_{top_p}_{speed}_{temperature}_{num_beams}.wav"
    
    try:
        # Tạo âm thanh
        final_wav = generate_full_audio(XTTS_MODEL, input_text, language="vi")
        
        # Lưu file bằng torchaudio (XTTS mặc định sample rate là 24000)
        torchaudio.save(output_filename, final_wav, sample_rate=24000)
        
        print(f"💾 Đã lưu file thành công: {os.path.abspath(output_filename)}")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
