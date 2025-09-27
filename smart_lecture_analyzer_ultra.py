#!/usr/bin/env python3
"""
SMART LECTURE ANALYZER ULTRA - Versione GPU Ultra-Ottimizzata 🚀
Qualità massima per trascrizione e analisi video con GPU

Miglioramenti Ultra:
- Whisper Large/Medium per massima accuratezza  
- Word-level timestamps precisi
- Frame rate ottimizzato per contenuto educativo
- OCR multi-threaded con preprocessing avanzato
- Rilevamento matematica/formule potenziato
- Output strutturato avanzato
"""

import whisper
import cv2
import pytesseract
import os
import sys
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime, timedelta
import re
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import torch

# Configurazione Tesseract per Windows
if sys.platform.startswith('win'):
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

# Forza l'uso della GPU e genera un errore se non disponibile
if not torch.cuda.is_available():
    raise RuntimeError("La GPU non è disponibile. Assicurati che CUDA sia installato e configurato correttamente.")

device = torch.device("cuda")
print("Esecuzione forzata su GPU: ", torch.cuda.get_device_name(0))

class SmartLectureAnalyzerUltra:
    def __init__(self, video_path, output_dir=None, quality_mode="ultra"):
        self.video_path = video_path
        self.base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = output_dir or os.path.dirname(video_path)
        self.quality_mode = quality_mode
        
        # File di stato per ripresa
        self.state_file = os.path.join(self.output_dir, f"{self.base_name}_state_ultra.pkl")
        
        print("🎤 Caricamento modello Whisper Ultra...")
        
        # Configurazione GPU ottimale per QUALITÀ MASSIMA
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU rilevata: {gpu_name}")
            print(f"💾 Memoria GPU: {gpu_memory:.1f} GB")
            
            # SEMPRE IL MEGLIO DISPONIBILE per qualità massima
            if gpu_memory >= 8:
                self.model_size = "large-v3"
                print("🏆 QUALITÀ MASSIMA: Whisper Large-v3")
            elif gpu_memory >= 6:
                self.model_size = "large-v2"  
                print("🥈 ALTA QUALITÀ: Whisper Large-v2")
            elif gpu_memory >= 4:
                self.model_size = "medium"  
                print("🥉 BUONA QUALITÀ: Whisper Medium")
            else:
                self.model_size = "base"
                print("✅ QUALITÀ BASE: Whisper Base")
        else:
            # Su CPU, usiamo medium se possibile per migliore qualità
            self.model_size = "medium"
            print("💻 CPU: Usando Whisper Medium per migliore qualità")
            
        self.whisper_model = whisper.load_model(self.model_size, device=self.device)
        
        # Configurazioni ULTRA-QUALITÀ ottimizzate per lo studio
        self.ultra_config = {
            "frame_interval": 10.0,  # Frame ogni 10 secondi (ultra-denso)
            "max_frames": 300,       # Fino a 300 frame (massimo dettaglio)
            "ocr_threads": 6,        # OCR più threads per qualità
            "math_detection": True,  # Rilevamento formule matematiche
            "enhance_images": True,  # Preprocessing avanzato immagini
            "word_timestamps": True, # Timestamp a livello parola
            "language_detection": True,  # Rilevamento lingua automatico
            "temperature": 0,        # Deterministic per consistenza
            "best_of": 5,           # Migliore di 5 tentativi
            "beam_size": 5,         # Beam search per migliore accuratezza
            "patience": 2.0         # Maggiore pazienza per qualità
        }
        
        # Strutture dati
        self.transcript_segments = []
        self.word_level_data = []
        self.visual_content = []
        self.mathematical_content = []
        self.scene_changes = []
        self.progress_state = {}
        
    def save_state(self):
        """Salva stato con dati estesi"""
        state = {
            'transcript_segments': self.transcript_segments,
            'word_level_data': self.word_level_data,
            'visual_content': self.visual_content,
            'mathematical_content': self.mathematical_content,
            'scene_changes': self.scene_changes,
            'progress_state': self.progress_state
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        """Carica stato con controllo compatibilità"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.transcript_segments = state.get('transcript_segments', [])
                    self.word_level_data = state.get('word_level_data', [])
                    self.visual_content = state.get('visual_content', [])
                    self.mathematical_content = state.get('mathematical_content', [])
                    self.scene_changes = state.get('scene_changes', [])
                    self.progress_state = state.get('progress_state', {})
                print("📂 Stato Ultra precedente caricato")
                return True
            except Exception as e:
                print(f"⚠️ Errore caricamento stato: {e}, ripartendo da zero")
        return False

    def extract_audio_transcript_ultra(self):
        """Trascrizione ultra-qualità con word timestamps"""
        if self.progress_state.get('transcript_done'):
            print("✅ Trascrizione Ultra già completata")
            return
            
        print("🎵 Trascrizione Ultra-Qualità in corso...")
        print(f"🔧 Modello: {self.model_size} su {self.device.upper()}")
        
        try:
            # Configurazione ultra per massima qualità
            result = self.whisper_model.transcribe(
                self.video_path,
                verbose=True,
                language=None,           # Auto-detect per massima flessibilità
                word_timestamps=self.ultra_config["word_timestamps"],
                fp16=True if self.device == "cuda" else False,  # FP16 su GPU per velocità
                temperature=0.0,         # Deterministic per consistenza
                compression_ratio_threshold=2.4,  # Migliore qualità audio
                logprob_threshold=-1.0,  # Più selettivo
                no_speech_threshold=0.6, # Migliore rilevamento silenzio
                condition_on_previous_text=True,  # Contesto per accuratezza
            )
            
            self.transcript_segments = result["segments"]
            self.full_transcript = result["text"]
            
            # Estrai word-level data se disponibili
            if self.ultra_config["word_timestamps"]:
                self.word_level_data = []
                for segment in self.transcript_segments:
                    if "words" in segment:
                        for word in segment["words"]:
                            self.word_level_data.append({
                                'word': word['word'],
                                'start': word['start'],
                                'end': word['end'],
                                'confidence': word.get('probability', 1.0)
                            })
            
            self.progress_state['transcript_done'] = True
            self.save_state()
            
            print(f"✅ Trascrizione Ultra completata:")
            print(f"   📊 {len(self.transcript_segments)} segmenti")
            print(f"   🔤 {len(self.word_level_data)} parole con timestamp")
            print(f"   🎯 Modello: {self.model_size}")
            
        except KeyboardInterrupt:
            print("🔄 Trascrizione interrotta, stato salvato")
            self.save_state()
            raise
        except Exception as e:
            print(f"❌ Errore trascrizione Ultra: {e}")
            raise

    def enhance_frame_for_ocr(self, frame):
        """Preprocessing avanzato frame per OCR ottimale"""
        if not self.ultra_config["enhance_images"]:
            return frame
            
        # Converti a PIL per processing avanzato
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Migliora contrasto e nitidezza
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Riduzione noise
        pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
        
        # Ritorna a OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def detect_mathematical_content(self, text):
        """Rilevamento formule e contenuto matematico"""
        if not self.ultra_config["math_detection"]:
            return False
            
        math_patterns = [
            r'[=<>≤≥≠±∑∏∫∂∇]',  # Simboli matematici
            r'\b\d+[.]\d+\b',      # Numeri decimali
            r'\b[xyz]\s*[=<>]',    # Variabili con operatori
            r'max|min|subject|s\.?t\.?',  # Terminologia ottimizzazione
            r'\\[a-zA-Z]+',        # Comandi LaTeX
            r'\$.*?\$',            # Math mode LaTeX
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def process_frame_ocr_ultra(self, frame_info):
        """OCR ultra-qualità con rilevamento matematico"""
        frame_path = frame_info['frame_path']
        timestamp = frame_info['timestamp']
        
        try:
            # Carica e preprocessa frame
            frame = cv2.imread(frame_path)
            enhanced_frame = self.enhance_frame_for_ocr(frame)
            
            # OCR multi-configurazione per massima accuratezza
            configs = [
                '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz=<>≤≥±∑∏∫∂()[]{}.,;:!?-+*/\\',
                '--psm 4',  # Single column
                '--psm 3',  # Fully automatic
            ]
            
            best_text = ""
            max_confidence = 0
            
            for config in configs:
                try:
                    # Estrai testo con confidenza
                    data = pytesseract.image_to_data(enhanced_frame, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calcola confidenza media
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        text = ' '.join([data['text'][i] for i in range(len(data['text'])) if int(data['conf'][i]) > 30])
                        
                        if avg_confidence > max_confidence and len(text.strip()) > 5:
                            max_confidence = avg_confidence
                            best_text = text
                            
                except Exception:
                    continue
            
            # Analizza contenuto matematico
            has_math = self.detect_mathematical_content(best_text)
            
            result = {
                'timestamp': timestamp,
                'text': best_text.strip(),
                'confidence': max_confidence,
                'has_math': has_math,
                'frame_path': frame_path
            }
            
            if has_math and best_text.strip():
                self.mathematical_content.append(result)
                
            return result
            
        except Exception as e:
            return {
                'timestamp': timestamp,
                'text': '',
                'confidence': 0,
                'has_math': False,
                'frame_path': frame_path,
                'error': str(e)
            }

    def extract_key_frames_ultra(self):
        """Estrazione frame ultra-densa con qualità ottimale"""
        if self.progress_state.get('frames_done'):
            print("✅ Estrazione frame Ultra già completata")
            return
            
        print("🖼️ Estrazione frame Ultra-Qualità...")
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            print(f"📹 Video: {duration:.1f} min, {fps:.1f} FPS")
            
            # Intervallo ottimizzato per contenuto educativo
            interval = self.ultra_config["frame_interval"]  # 15 secondi
            max_frames = self.ultra_config["max_frames"]     # 200 frame max
            
            # Se video molto lungo, adatta intervallo
            estimated_frames = int(duration / interval)
            if estimated_frames > max_frames:
                interval = duration / max_frames
                
            print(f"🎯 Estrarrò ~{int(duration/interval)} frame ogni {interval:.1f}s")
            
            # Genera timestamp per estrazione
            frame_times = []
            current_time = 0
            while current_time < duration:
                frame_times.append(current_time)
                current_time += interval
            
            frames_dir = os.path.join(self.output_dir, f"{self.base_name}_frames_ultra")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Estrai frame con qualità ottimale
            for i, timestamp in enumerate(frame_times):
                frame_num = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    # Salva con qualità massima
                    frame_path = os.path.join(frames_dir, f"frame_{i:04d}_{timestamp:.1f}s.jpg")
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    self.visual_content.append({
                        'timestamp': timestamp,
                        'frame_path': frame_path,
                        'frame_number': frame_num,
                        'frame_index': i
                    })
            
            cap.release()
            self.progress_state['frames_done'] = True
            self.save_state()
            
            print(f"✅ Estratti {len(self.visual_content)} frame Ultra")
            
        except Exception as e:
            print(f"❌ Errore estrazione frame Ultra: {e}")
            self.save_state()

    def process_all_frames_ultra(self):
        """Elaborazione parallela di tutti i frame con OCR"""
        if self.progress_state.get('ocr_done'):
            print("✅ OCR Ultra già completato")
            return
            
        print("🔍 OCR Ultra-Qualità su tutti i frame...")
        
        # Processa frame in parallelo per velocità
        with ThreadPoolExecutor(max_workers=self.ultra_config["ocr_threads"]) as executor:
            ocr_results = list(executor.map(self.process_frame_ocr_ultra, self.visual_content))
        
        # Aggiorna visual_content con risultati OCR
        for i, result in enumerate(ocr_results):
            self.visual_content[i].update(result)
            
        self.progress_state['ocr_done'] = True
        self.save_state()
        
        # Statistiche
        texts_found = sum(1 for r in ocr_results if r['text'])
        math_found = sum(1 for r in ocr_results if r['has_math'])
        
        print(f"✅ OCR Ultra completato:")
        print(f"   📝 {texts_found} frame con testo")
        print(f"   🧮 {math_found} frame con contenuto matematico")

    def analyze_complete_ultra(self):
        """Analisi completa ultra-qualità"""
        print(f"🚀 AVVIO ANALISI ULTRA: {self.base_name}")
        print("=" * 60)
        
        # Carica stato precedente
        self.load_state()
        
        try:
            # 1. Trascrizione Ultra
            self.extract_audio_transcript_ultra()
            
            # 2. Estrazione frame Ultra
            self.extract_key_frames_ultra()
            
            # 3. OCR Ultra parallelo
            self.process_all_frames_ultra()
            
            # 4. Report Ultra
            report_path = self.generate_ultra_report()
            
            # Cleanup
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                
            return report_path
            
        except KeyboardInterrupt:
            print("🔄 Analisi Ultra interrotta, progresso salvato")
            return None
        except Exception as e:
            print(f"❌ Errore analisi Ultra: {e}")
            return None

    def generate_ultra_report(self):
        """Report ultra-dettagliato con tutte le funzionalità"""
        print("📝 Generazione Report Ultra...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"{self.base_name}_ultra_notes_{timestamp}.md")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header Ultra
            f.write(f"# 🚀 SMART LECTURE ANALYZER ULTRA REPORT\n\n")
            f.write(f"**File:** {self.base_name}\n")
            f.write(f"**Processato:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Sistema:** Ultra GPU-Optimized\n")
            f.write(f"**Modello Whisper:** {self.model_size} su {self.device.upper()}\n")
            f.write(f"**Segmenti audio:** {len(self.transcript_segments)}\n")
            f.write(f"**Word timestamps:** {len(self.word_level_data)}\n")
            f.write(f"**Frame estratti:** {len(self.visual_content)}\n")
            f.write(f"**Contenuto matematico:** {len(self.mathematical_content)} frame\n\n")
            f.write("---\n\n")
            
            # Sezione contenuto matematico se presente
            if self.mathematical_content:
                f.write("## 🧮 CONTENUTO MATEMATICO RILEVATO\n\n")
                for item in self.mathematical_content[:10]:  # Top 10
                    time_str = f"{int(item['timestamp']//60):02d}:{int(item['timestamp']%60):02d}"
                    f.write(f"**[{time_str}]** Confidenza: {item['confidence']:.1f}%\n")
                    f.write(f"```\n{item['text']}\n```\n")
                    f.write(f"*Frame: {os.path.basename(item['frame_path'])}*\n\n")
                f.write("---\n\n")
            
            # Indice ultra-dettagliato
            f.write("## 📚 INDICE TEMPORALE ULTRA\n\n")
            for segment in self.transcript_segments[:15]:  # Prime 15 per anteprima
                start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
                preview = segment['text'][:100].strip()
                f.write(f"- **{start_time}**: {preview}...\n")
            f.write("\n---\n\n")
            
            # Trascrizione completa con word timestamps
            f.write("## 🎵 TRASCRIZIONE ULTRA-DETTAGLIATA\n\n")
            for segment in self.transcript_segments:
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
                
                start_time = f"{int(start//60):02d}:{int(start%60):02d}"
                end_time = f"{int(end//60):02d}:{int(end%60):02d}"
                
                f.write(f"### [{start_time} - {end_time}]\n\n")
                f.write(f"**Audio:**\n")
                f.write(f"> {text}\n\n")
                
                # Aggiungi frame corrispondenti se disponibili
                relevant_frames = [
                    frame for frame in self.visual_content 
                    if start <= frame['timestamp'] <= end and frame.get('text', '').strip()
                ]
                
                if relevant_frames:
                    f.write("**Contenuto visivo:**\n")
                    for frame in relevant_frames[:2]:  # Max 2 per segment
                        if frame.get('text'):
                            f.write(f"- *Frame {frame['timestamp']:.1f}s*: {frame['text'][:80]}...\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        print(f"✅ Report Ultra salvato: {output_path}")
        return output_path

    def detect_and_correct_repetitions(self, transcription):
        """Rileva e corregge ripetizioni nella trascrizione."""
        corrected_transcription = []
        for line in transcription:
            if corrected_transcription and line.strip() == corrected_transcription[-1].strip():
                print(f"Rilevata ripetizione: '{line.strip()}' - Ignorata.")
                continue
            corrected_transcription.append(line)
        return corrected_transcription

    def remove_duplicates(self, transcription):
        """Rimuove righe duplicate consecutive dalla trascrizione."""
        filtered_transcription = []
        for line in transcription:
            if not filtered_transcription or line != filtered_transcription[-1]:
                filtered_transcription.append(line)
        return filtered_transcription

def main():
    if len(sys.argv) < 2:
        print("Uso: python smart_lecture_analyzer_ultra.py <video_file> [quality_mode]")
        print("Quality modes: ultra (default), high, standard")
        sys.exit(1)
    
    video_path = sys.argv[1]
    quality_mode = sys.argv[2] if len(sys.argv) > 2 else "ultra"
    
    if not os.path.exists(video_path):
        print(f"❌ File non trovato: {video_path}")
        sys.exit(1)
    
    try:
        analyzer = SmartLectureAnalyzerUltra(video_path, quality_mode=quality_mode)
        result = analyzer.analyze_complete_ultra()
        
        if result:
            print(f"🎉 ANALISI ULTRA COMPLETATA! Report: {result}")
        else:
            print("⚠️ Analisi interrotta o fallita")
            
    except KeyboardInterrupt:
        print("\n🔄 Processo interrotto dall'utente")
    except Exception as e:
        print(f"❌ Errore generale: {e}")

if __name__ == "__main__":
    main()