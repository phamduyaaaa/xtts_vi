import time
import torch
import numpy as np
import sounddevice as sd
from underthesea import sent_tokenize
from vinorm import TTSnorm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH (OFFLINE 100%)
# ==========================================
print("--- [1] ĐANG KHỞI TẠO HỆ THỐNG TỪ LOCAL ---")

# Đường dẫn cứng tới file model nội bộ (Không check mạng)
MODEL_DIR = "model/"
config_file = f"{MODEL_DIR}config.json"
model_weights = f"{MODEL_DIR}model.pth"
vocab_file = f"{MODEL_DIR}vocab.json"
speaker_audio_file = f"{MODEL_DIR}vi_man.wav"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model thẳng vào VRAM
config = XttsConfig()
config.load_json(config_file)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config,
                            checkpoint_path=model_weights,
                            vocab_path=vocab_file,
                            use_deepspeed=False)
XTTS_MODEL.to(device)

# Load Latent Vector của giọng mẫu
gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)
print("--- KHỞI TẠO XONG. SẴN SÀNG STREAMING! ---\n")

# ==========================================
# 2. TIỀN XỬ LÝ (CHUNKING)
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
            
            # Ép chunk siêu nhỏ (dưới 10 từ) để trả FPL lẹ nhất
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
# 4. VÒNG LẶP TƯƠNG TÁC & ĐO LƯỜNG FPL
# ==========================================
if __name__ == "__main__":
    print("======================================================")
    print("🎤 INTERACTIVE MODE - ĐO LƯỜNG ĐỘ TRỄ FPL")
    print("======================================================\n")
    
    while True:
        try:
            input_text = input("✍️ Nhập văn bản (Gõ 'exit' để thoát): ")
            
            if input_text.lower() in ['exit', 'quit']:
                break
            if not input_text.strip():
                continue
            
            # -----------------------------------------------------
            # BẮT ĐẦU BẤM GIỜ NGAY SAU KHI USER NHẤN ENTER
            # -----------------------------------------------------
            start_time = time.time()
            is_first_chunk = True
            
            # Mở luồng audio
            audio_stream = sd.OutputStream(samplerate=24000, channels=1, dtype='float32')
            audio_stream.start()
            
            print("⏳ Đang xử lý...", end="", flush=True)
            
            try:
                for audio_tensor in tts_streaming(XTTS_MODEL, input_text, "vi", gpt_cond_latent, speaker_embedding):
                    
                    # NẾU LÀ CHUNK ĐẦU TIÊN -> CHỐT THỜI GIAN FPL
                    if is_first_chunk:
                        fpl_time = time.time() - start_time
                        print(f"\n🚀 TỐC ĐỘ PHẢN HỒI (FPL): {fpl_time:.3f} giây!")
                        print("🔊 Đang phát:", end=" ", flush=True)
                        is_first_chunk = False
                    
                    print("■", end="", flush=True) # Visualizer luồng stream
                    
                    # Phát âm thanh ra loa
                    audio_np = audio_tensor.squeeze().cpu().numpy()
                    audio_stream.write(audio_np)
                    
            finally:
                audio_stream.stop()
                audio_stream.close()
                
                total_time = time.time() - start_time
                print(f"\n⏱️ Tổng thời gian xử lý & phát: {total_time:.2f}s\n")
                
        except KeyboardInterrupt:
            print("\n[!] Đã dừng phát. Gõ 'exit' để thoát.")
