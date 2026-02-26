import time
import torch
import queue
import threading
import numpy as np
import sounddevice as sd
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH (MAX OPTIMIZATION)
# ==========================================
print("--- [1] ĐANG KHỞI TẠO HỆ THỐNG ---")

MODEL_DIR = "model/"
config_file = f"{MODEL_DIR}config.json"
model_weights = f"{MODEL_DIR}model.pth"
vocab_file = f"{MODEL_DIR}vocab.json"
speaker_audio_file = f"{MODEL_DIR}vi_man.wav"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

config = XttsConfig()
config.load_json(config_file)
XTTS_MODEL = Xtts.init_from_config(config)

# Tối ưu DeepSpeed & FP16
try:
    import deepspeed
    use_deepspeed = True if device.startswith("cuda") else False
except ImportError:
    use_deepspeed = False

XTTS_MODEL.load_checkpoint(config,
                            checkpoint_path=model_weights,
                            vocab_path=vocab_file,
                            use_deepspeed=use_deepspeed)

if device.startswith("cuda"):
    XTTS_MODEL.half()

XTTS_MODEL.to(device)

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

if device.startswith("cuda"):
    gpt_cond_latent = gpt_cond_latent.half()
    speaker_embedding = speaker_embedding.half()

print(f"--- KHỞI TẠO XONG. DEEPSPEED: {use_deepspeed} | DEVICE: {device.upper()} ---\n")

# ==========================================
# 2. CORE GENERATOR (NATIVE STREAMING)
# ==========================================
def tts_streaming_native(model, text, language, gpt_cond_latent, speaker_embedding):
    """
    Nhận TRỌN BỘ câu văn, model tự động tính toán bối cảnh và yield ra từng frame audio.
    Không dùng hàm cắt text (chunking) nữa.
    """
    with torch.inference_mode():
        # Sử dụng hàm inference_stream thay vì inference
        audio_stream = model.inference_stream(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            enable_text_chunking=False # Ép model không tự ý băm text để giữ 100% ngữ cảnh
        )
        
        for audio_chunk in audio_stream:
            yield audio_chunk

# ==========================================
# 3. CONSUMER THREAD (PHÁT LOA ĐỘC LẬP)
# ==========================================
def audio_playback_worker(audio_queue):
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype='float32')
    stream.start()
    while True:
        audio_np = audio_queue.get()
        if audio_np is None:
            break
        stream.write(audio_np)
        audio_queue.task_done()
    stream.stop()
    stream.close()

# ==========================================
# 4. VÒNG LẶP TƯƠNG TÁC
# ==========================================
if __name__ == "__main__":
    print("======================================================")
    print("🎤 NATIVE STREAMING MODE (KHÔNG CẮT CHUNK)")
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
            
            audio_queue = queue.Queue()
            player_thread = threading.Thread(target=audio_playback_worker, args=(audio_queue,))
            player_thread.start()
            
            print("⏳ Đang phân tích toàn bộ bối cảnh câu...", end="", flush=True)
            
            try:
                # Đưa nguyên 100% text vào
                for audio_tensor in tts_streaming_native(XTTS_MODEL, input_text, "vi", gpt_cond_latent, speaker_embedding):
                    
                    if is_first_chunk:
                        fpl_time = time.time() - start_time
                        print(f"\n🚀 FPL (Độ trễ gói đầu tiên): {fpl_time:.3f} giây!")
                        print("🔊 Đang phát:", end=" ", flush=True)
                        is_first_chunk = False
                    
                    print("■", end="", flush=True)
                    
                    # XTTS trả về Tensor hoặc Numpy tùy phiên bản, bọc thêm hàm tensor để an toàn
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_np = audio_tensor.squeeze().cpu().numpy()
                    else:
                        audio_np = np.array(audio_tensor).squeeze()
                        
                    audio_queue.put(audio_np)
                    
            finally:
                audio_queue.put(None)
                player_thread.join()
                
                total_time = time.time() - start_time
                print(f"\n⏱️ Tổng thời gian (Xử lý + Phát xong): {total_time:.2f}s\n")
                
        except KeyboardInterrupt:
            print("\n[!] Đã dừng phát.")
            try:
                audio_queue.put(None)
                player_thread.join(timeout=1)
            except:
                pass
