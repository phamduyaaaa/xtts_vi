import time
import torch
import queue
import threading
import numpy as np
import sounddevice as sd
from underthesea import sent_tokenize
from vinorm import TTSnorm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH (MAX OPTIMIZATION LEVEL)
# ==========================================
print("--- [1] ĐANG KHỞI TẠO HỆ THỐNG (TỐI ƯU HÓA) ---")

MODEL_DIR = "model/"
config_file = f"{MODEL_DIR}config.json"
model_weights = f"{MODEL_DIR}model.pth"
vocab_file = f"{MODEL_DIR}vocab.json"
speaker_audio_file = f"{MODEL_DIR}vi_man.wav"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

config = XttsConfig()
config.load_json(config_file)
XTTS_MODEL = Xtts.init_from_config(config)

# [TỐI ƯU 1]: Bật DeepSpeed nếu có GPU để tăng tốc độ nhân ma trận
try:
    import deepspeed
    use_deepspeed = True if device.startswith("cuda") else False
except ImportError:
    print("[!] Cảnh báo: Chưa cài DeepSpeed. Chạy `pip install deepspeed` để tăng tốc độ.")
    use_deepspeed = False

XTTS_MODEL.load_checkpoint(config,
                            checkpoint_path=model_weights,
                            vocab_path=vocab_file,
                            use_deepspeed=use_deepspeed)

# [TỐI ƯU 2]: Ép kiểu dữ liệu về FP16 (Half-precision) giúp GPU chạy nhanh gấp đôi
if device.startswith("cuda"):
    XTTS_MODEL.half()

XTTS_MODEL.to(device)

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

# Chuyển vector điều kiện sang cùng kiểu dữ liệu với model (FP16)
if device.startswith("cuda"):
    gpt_cond_latent = gpt_cond_latent.half()
    speaker_embedding = speaker_embedding.half()

print(f"--- KHỞI TẠO XONG. DEEPSPEED: {use_deepspeed} | DEVICE: {device.upper()} ---\n")

# ==========================================
# 2. TIỀN XỬ LÝ (CHUNKING) - Giữ nguyên logic cũ
# ==========================================
def preprocess_text_for_streaming(text, language="vi"):
    if language == "vi":
        text = TTSnorm(text, unknown=False, lower=False, rule=True)
    sentences = sent_tokenize(text)
    chunks = []
    chunk_i = ""
    len_chunk_i = 0
    for sentence in sentences:
        sub_sentences = sentence.split(',') 
        for sub in sub_sentences:
            sub = sub.strip()
            if not sub: continue
            chunk_i += " " + sub + ","
            len_chunk_i += len(sub.split())
            if len_chunk_i >= 8: 
                chunks.append(chunk_i.strip())
                chunk_i = ""
                len_chunk_i = 0
    if chunk_i.strip():
        chunks.append(chunk_i.strip())
    return [c.rstrip(',') for c in chunks if c]

# ==========================================
# 3. CORE GENERATOR
# ==========================================
def tts_streaming(model, text, language, gpt_cond_latent, speaker_embedding):
    chunks = preprocess_text_for_streaming(text, language)
    # [TỐI ƯU 3]: Sử dụng torch.inference_mode() nhanh hơn torch.no_grad()
    with torch.inference_mode():
        for text_chunk in chunks:
            if text_chunk.strip() == "": continue
            wav_chunk = model.inference(
                text=text_chunk,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=10,
                top_p=0.5,
            )
            yield torch.tensor(wav_chunk["wav"])

# ==========================================
# 4. [TỐI ƯU 4]: LUỒNG PHÁT ÂM THANH ĐỘC LẬP (CONSUMER THREAD)
# ==========================================
def audio_playback_worker(audio_queue):
    """Luồng này chỉ chuyên đọc từ Queue và nhét vào loa, không làm phiền GPU"""
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype='float32')
    stream.start()
    while True:
        audio_np = audio_queue.get()
        if audio_np is None: # Tín hiệu dừng luồng (Poison pill)
            break
        stream.write(audio_np)
        audio_queue.task_done()
    stream.stop()
    stream.close()

# ==========================================
# 5. VÒNG LẶP TƯƠNG TÁC CHÍNH (PRODUCER)
# ==========================================
if __name__ == "__main__":
    print("======================================================")
    print("🎤 INTERACTIVE MODE - MAX SPEED OPTIMIZED")
    print("======================================================\n")
    
    while True:
        try:
            input_text = input("✍️ Nhập văn bản (Gõ 'exit' để thoát): ")
            if input_text.lower() in ['exit', 'quit']:
                break
            if not input_text.strip():
                continue
            
            start_time = time.time()
            is_first_chunk = True
            
            # Khởi tạo Hàng đợi (Queue) và khởi động Luồng phát âm thanh (Consumer)
            audio_queue = queue.Queue()
            player_thread = threading.Thread(target=audio_playback_worker, args=(audio_queue,))
            player_thread.start()
            
            print("⏳ Đang xử lý...", end="", flush=True)
            
            try:
                # GPU chỉ lo việc sinh audio và ném vào Queue se rất nhanh
                for audio_tensor in tts_streaming(XTTS_MODEL, input_text, "vi", gpt_cond_latent, speaker_embedding):
                    
                    if is_first_chunk:
                        fpl_time = time.time() - start_time
                        print(f"\n🚀 FPL (Độ trễ gói đầu tiên): {fpl_time:.3f} giây!")
                        print("🔊 Đang phát:", end=" ", flush=True)
                        is_first_chunk = False
                    
                    print("■", end="", flush=True)
                    
                    # Ném array numpy vào Queue cho luồng kia tự lo việc đọc
                    audio_np = audio_tensor.squeeze().cpu().numpy()
                    audio_queue.put(audio_np)
                    
            finally:
                # Đợi phát hết audio trong Queue rồi gửi tín hiệu dừng (None)
                audio_queue.put(None)
                player_thread.join() # Chờ luồng loa tắt hẳn
                
                total_time = time.time() - start_time
                print(f"\n⏱️ Tổng thời gian (Xử lý + Phát xong): {total_time:.2f}s\n")
                
        except KeyboardInterrupt:
            print("\n[!] Đã dừng phát.")
            # Xử lý dọn dẹp luồng nếu user bấm Ctrl+C
            try:
                audio_queue.put(None)
                player_thread.join(timeout=1)
            except:
                pass
