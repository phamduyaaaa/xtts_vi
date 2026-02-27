'''
Code chay inference khi da chay save voice
Khong su dung ttsnorm
Tích hợp Smart Chunking Generator
Phát trực tiếp ra loa (Live Streaming Playback)
'''

import os
import time
import re
import torch
import numpy as np
import sounddevice as sd
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ==========================================
# 1. CẤU HÌNH THAM SỐ (ĐÃ TỐI ƯU HẾT VỠ TIẾNG)
# ==========================================
length_penalty = 1.0         # Chuẩn độ dài
repetition_penalty = 2.5     # Ngăn lặp từ nhưng không làm rè/vỡ tiếng
top_k = 50                   # Tăng lựa chọn âm tự nhiên
top_p = 0.8                  # Độ biến thiên cảm xúc tốt
speed = 0.85                 # Đọc chậm, từ tốn
temperature = 0.7            # Cân bằng giữa biểu cảm và ổn định
num_beams = 1                # BẮT BUỘC ĐỂ 1 để tránh tiếng robot

MODEL_DIR = "model/"
config_file = f"{MODEL_DIR}config.json"
model_weights = f"{MODEL_DIR}model.pth"
vocab_file = f"{MODEL_DIR}vocab.json"
latents_file = f"{MODEL_DIR}vi_man_latents.pth" # File Caching đã tạo ở Bước 1

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH (SIÊU TỐC)
# ==========================================
print("--- [1] ĐANG KHỞI TẠO HỆ THỐNG ---")
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

# LOAD CACHE GIỌNG NÓI (Bỏ qua bước trích xuất wav chậm chạp)
print(f"[*] Đang load đặc trưng giọng nói từ: {latents_file}")
if not os.path.exists(latents_file):
    raise FileNotFoundError(f"LỖI: Không tìm thấy {latents_file}. Hãy chạy file save_voice.py trước!")

# Load file .pth và đẩy thẳng vào GPU/CPU
latents = torch.load(latents_file, map_location=device, weights_only=True)
gpt_cond_latent = latents["gpt_cond_latent"].to(device)
speaker_embedding = latents["speaker_embedding"].to(device)

print("--- KHỞI TẠO XONG! HỆ THỐNG ĐÃ SẴN SÀNG --- \n")

# ==========================================
# 3. LOGIC SMART CHUNKING (CẮT CÂU THÔNG MINH)
# ==========================================
def split_text_smartly(text, min_words=5):
    """
    Cắt đoạn văn dài thành các mảnh nhỏ dựa trên dấu câu (.,!?;).
    Ghép các mảnh quá ngắn lại với nhau để đảm bảo ngữ điệu.
    """
    phrases = re.split(r'([.,!?;])', text)
    chunks = []
    current_chunk = ""

    for i in range(0, len(phrases) - 1, 2):
        phrase = phrases[i].strip()
        punct = phrases[i+1].strip()
        if not phrase: continue

        current_chunk += phrase + punct + " "
        
        if len(current_chunk.split()) >= min_words:
            chunks.append(current_chunk.strip())
            current_chunk = ""

    if len(phrases) % 2 != 0 and phrases[-1].strip():
        current_chunk += phrases[-1].strip()

    if current_chunk.strip():
        if chunks: 
            chunks[-1] += " " + current_chunk.strip()
        else:
            chunks.append(current_chunk.strip())

    return chunks

# ==========================================
# 4. HÀM TỔNG HỢP (GENERATOR)
# ==========================================
def generate_audio_smart_stream(model, text, language="vi"):
    chunks = split_text_smartly(text)
    print(f"🔪 Đã tự động băm văn bản thành {len(chunks)} Chunks nhỏ để Streaming.\n")
    
    with torch.inference_mode():
        for i, chunk_text in enumerate(chunks):
            print(f"⏳ Đang xử lý Chunk {i+1}/{len(chunks)}: '{chunk_text}'")
            start_time = time.time()
            
            outputs = model.inference(
                text=chunk_text,
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
            print(f"   ✅ Xử lý xong trong: {end_time - start_time:.2f} giây (Bắt đầu phát ra loa...)")
            
            # Chuyển đổi sang Tensor và trả về từng phần (Yield)
            audio_tensor = torch.tensor(outputs["wav"]).unsqueeze(0)
            yield audio_tensor

# ==========================================
# 5. THỰC THI (PHÁT TRỰC TIẾP RA LOA)
# ==========================================
if __name__ == "__main__":
    input_text = (
        "Ôm chặt tấm hình nhỏ trong tay, người cha lặng nhìn căn phòng trống nơi tiếng cười con từng vang lên mỗi buổi chiều. "
        "Ngoài kia, cơn mưa vẫn rơi hòa cùng những giọt nước mắt chẳng thể ngừng. Ông khẽ thì thầm. Ba về rồi đây con ơi, ba mang bánh về cho con đây. "
        "Nhưng chỉ còn lại khoảng lặng, và lời hứa mãi dang dở giữa hai bờ yêu thương."
    )
    
    print("🔊 Đang kết nối với loa hệ thống...")
    # Khởi tạo luồng loa: XTTS sinh ra âm thanh Mono (1 kênh), sample rate 24kHz, định dạng float32
    audio_stream = sd.OutputStream(samplerate=24000, channels=1, dtype='float32')
    audio_stream.start()
    
    try:
        # Nhận từng mảnh âm thanh được yield ra từ Generator và bơm thẳng vào loa
        for audio_chunk in generate_audio_smart_stream(XTTS_MODEL, input_text, language="vi"):
            # Ép kiểu Tensor về Numpy Array 1 chiều
            audio_np = audio_chunk.squeeze().cpu().numpy()
            
            # Đẩy dữ liệu vào loa (Hàm write sẽ tự block chờ loa đọc nếu âm thanh đang phát chưa hết)
            audio_stream.write(audio_np)
            
        print("\n✅ Đã đọc xong toàn bộ văn bản!")
        
    except KeyboardInterrupt:
        print("\n[!] Đã ép dừng phát âm thanh.")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
    finally:
        # Dọn dẹp đóng luồng loa
        audio_stream.stop()
        audio_stream.close()
