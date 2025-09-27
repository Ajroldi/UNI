#!/usr/bin/env python3
"""
Script completo per tradurre file Markdown e arricchirli con contenuto video
Utilizza risorse open source gratuite con supporto GPU NVIDIA
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import json

# Librerie per video processing
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Librerie per traduzione (usando GPU)
try:
    from transformers import MarianMTModel, MarianTokenizer, pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Librerie per OCR
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Librerie per screenshot
try:
    import pyautogui
    HAS_SCREENSHOT = True
except ImportError:
    HAS_SCREENSHOT = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GPUTranslator:
    """Traduttore che sfrutta GPU NVIDIA per massime prestazioni"""
    
    def __init__(self, source_lang: str = "en", target_lang: str = "it"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Utilizzando dispositivo: {self.device}")
        
        if self.device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU disponibile: {gpu_name}")
            except:
                logger.info("GPU CUDA disponibile")
        
        # Modello per traduzione EN->IT (Helsinki-NLP è eccellente e gratuito)
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        
        try:
            logger.info(f"Caricamento modello {model_name}...")
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            logger.info(f"Modello {model_name} caricato su {self.device}")
        except Exception as e:
            logger.error(f"Errore nel caricare il modello: {e}")
            try:
                # Fallback a pipeline più semplice
                self.translator = pipeline("translation", 
                                         model=model_name, 
                                         device=0 if self.device == "cuda" else -1)
                logger.info("Fallback a pipeline completato")
            except Exception as e2:
                logger.error(f"Errore anche nel fallback: {e2}")
                raise e2
    
    def translate_text(self, text: str, max_length: int = 512) -> str:
        """Traduce testo usando GPU"""
        if not text or not text.strip():
            return text
            
        try:
            # Pulisci il testo
            clean_text = text.strip()
            if len(clean_text) == 0:
                return text
                
            # Spezza testi lunghi in chunks
            chunks = self._split_text(clean_text, max_length)
            translated_chunks = []
            
            for chunk in chunks:
                if hasattr(self, 'model'):
                    # Usa modello diretto per massime prestazioni
                    inputs = self.tokenizer(chunk, return_tensors="pt", 
                                          padding=True, truncation=True, 
                                          max_length=max_length).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, 
                                                    max_length=max_length,
                                                    num_beams=4,
                                                    early_stopping=True)
                    
                    translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # Usa pipeline
                    result = self.translator(chunk)
                    translated = result[0]['translation_text']
                
                translated_chunks.append(translated.strip())
            
            return " ".join(translated_chunks)
            
        except Exception as e:
            logger.error(f"Errore traduzione per testo '{text[:50]}...': {e}")
            return text
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Divide il testo in chunks mantenendo la coerenza"""
        # Se il testo è già abbastanza corto, restituiscilo così com'è
        if len(text) <= max_length:
            return [text]
            
        # Dividi per frasi prima
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # Se anche una singola frase è troppo lunga, dividila per parole
            if sentence_length > max_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Dividi la frase lunga per parole
                words = sentence.split()
                word_chunk = []
                word_length = 0
                
                for word in words:
                    if word_length + len(word) > max_length and word_chunk:
                        chunks.append(" ".join(word_chunk))
                        word_chunk = [word]
                        word_length = len(word)
                    else:
                        word_chunk.append(word)
                        word_length += len(word) + 1
                
                if word_chunk:
                    chunks.append(" ".join(word_chunk))
                    
            elif current_length + sentence_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]

class OCRProcessor:
    """Processore OCR che combina EasyOCR (GPU) e Tesseract per massima accuratezza"""
    
    def __init__(self):
        self.easyocr_reader = None
        self.tesseract_available = HAS_TESSERACT
        
        if HAS_EASYOCR:
            try:
                # EasyOCR con GPU per migliori prestazioni
                gpu_enabled = torch.cuda.is_available() if HAS_TRANSFORMERS else False
                self.easyocr_reader = easyocr.Reader(['en', 'it'], gpu=gpu_enabled)
                logger.info(f"EasyOCR inizializzato con GPU: {gpu_enabled}")
            except Exception as e:
                logger.error(f"Errore inizializzazione EasyOCR: {e}")
                self.easyocr_reader = None
        
        if HAS_TESSERACT:
            # Configura Tesseract per massima qualità
            self.tesseract_config = '--oem 3 --psm 6'
            logger.info("Tesseract configurato per IT/EN")
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, str]:
        """Estrae testo da immagine usando tutti i motori OCR disponibili"""
        results = {}
        
        if not os.path.exists(image_path):
            logger.error(f"Immagine non trovata: {image_path}")
            return results
        
        try:
            # Preprocessa immagine per migliorare OCR
            processed_image = self._preprocess_image(image_path)
            
            # EasyOCR (migliore per testo multilingue)
            if self.easyocr_reader:
                try:
                    easyocr_result = self.easyocr_reader.readtext(processed_image)
                    easyocr_text = " ".join([item[1] for item in easyocr_result if item[2] > 0.3])
                    if easyocr_text.strip():
                        results['easyocr'] = easyocr_text.strip()
                        logger.info(f"EasyOCR estratto: {len(easyocr_text)} caratteri")
                except Exception as e:
                    logger.error(f"Errore EasyOCR: {e}")
            
            # Tesseract (ottimo per documenti strutturati)
            if self.tesseract_available:
                try:
                    pil_image = Image.fromarray(processed_image)
                    tesseract_text = pytesseract.image_to_string(pil_image, 
                                                               lang='eng+ita',
                                                               config=self.tesseract_config)
                    tesseract_clean = tesseract_text.strip()
                    if tesseract_clean and len(tesseract_clean) > 5:
                        results['tesseract'] = tesseract_clean
                        logger.info(f"Tesseract estratto: {len(tesseract_clean)} caratteri")
                except Exception as e:
                    logger.error(f"Errore Tesseract: {e}")
                    
        except Exception as e:
            logger.error(f"Errore generale OCR per {image_path}: {e}")
        
        return results
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocessa immagine per migliorare accuratezza OCR"""
        if not HAS_CV2:
            # Se non abbiamo OpenCV, usa PIL per caricare l'immagine
            pil_img = Image.open(image_path)
            return np.array(pil_img)
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossibile caricare immagine: {image_path}")
        
        # Converti in scala di grigi
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Migliora il contrasto
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Riduci rumore
        denoised = cv2.medianBlur(enhanced, 3)
        
        return denoised

class VideoProcessor:
    """Processore per estrarre frame da video MP4"""
    
    def __init__(self, output_dir: str = "video_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not HAS_CV2:
            logger.warning("OpenCV non disponibile - funzionalità video limitate")
    
    def extract_frames_from_video(self, video_path: str, interval_seconds: int = 60) -> List[str]:
        """Estrae frame dal video a intervalli regolari"""
        if not HAS_CV2:
            logger.error("OpenCV non disponibile per processamento video")
            return []
        
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video non trovato: {video_path}")
            return []
        
        extracted_frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Impossibile aprire video: {video_path}")
                return []
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                logger.error("FPS non valido nel video")
                cap.release()
                return []
                
            frame_interval = int(fps * interval_seconds)
            frame_count = 0
            saved_count = 0
            
            logger.info(f"Estraendo frame ogni {interval_seconds} secondi (FPS: {fps:.2f})")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    
                    frame_filename = f"{video_path.stem}_frame_{minutes:02d}m{seconds:02d}s.jpg"
                    frame_path = self.output_dir / frame_filename
                    
                    success = cv2.imwrite(str(frame_path), frame)
                    if success:
                        extracted_frames.append(str(frame_path))
                        saved_count += 1
                        logger.info(f"Frame salvato: {frame_filename} ({minutes:02d}:{seconds:02d})")
                    else:
                        logger.error(f"Errore salvataggio frame: {frame_filename}")
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Estratti {saved_count} frame da {video_path.name}")
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Errore estrazione frame: {e}")
            return []
    
    def extract_key_frames(self, video_path: str, threshold: float = 0.3) -> List[str]:
        """Estrae frame chiave basandosi su cambiamenti di scena"""
        if not HAS_CV2:
            logger.error("OpenCV non disponibile per rilevamento frame chiave")
            return []
        
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video non trovato: {video_path}")
            return []
            
        extracted_frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Impossibile aprire video: {video_path}")
                return []
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                logger.error("FPS non valido nel video")
                cap.release()
                return []
            
            ret, prev_frame = cap.read()
            if not ret:
                logger.error("Impossibile leggere primo frame")
                cap.release()
                return []
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_count = 0
            saved_count = 0
            
            # Salva primo frame
            frame_filename = f"{video_path.stem}_key_frame_{saved_count:03d}_00m00s.jpg"
            frame_path = self.output_dir / frame_filename
            if cv2.imwrite(str(frame_path), prev_frame):
                extracted_frames.append(str(frame_path))
                saved_count += 1
                logger.info(f"Frame chiave iniziale salvato: {frame_filename}")
            
            logger.info(f"Rilevando frame chiave (soglia: {threshold})")
            
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Calcola differenza tra frame consecutivi
                diff = cv2.absdiff(prev_gray, curr_gray)
                diff_score = np.mean(diff) / 255.0
                
                if diff_score > threshold:
                    timestamp = frame_count / fps
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    
                    frame_filename = f"{video_path.stem}_key_frame_{saved_count:03d}_{minutes:02d}m{seconds:02d}s.jpg"
                    frame_path = self.output_dir / frame_filename
                    
                    if cv2.imwrite(str(frame_path), curr_frame):
                        extracted_frames.append(str(frame_path))
                        saved_count += 1
                        logger.info(f"Frame chiave: {frame_filename} ({minutes:02d}:{seconds:02d}) - Diff: {diff_score:.3f}")
                
                prev_gray = curr_gray.copy()
                frame_count += 1
            
            cap.release()
            logger.info(f"Estratti {saved_count} frame chiave da {video_path.name}")
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Errore estrazione frame chiave: {e}")
            return []
    
    def get_video_info(self, video_path: str) -> Dict:
        """Ottiene informazioni sul video"""
        if not HAS_CV2:
            return {"error": "OpenCV non disponibile"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Impossibile aprire video"}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            info = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': duration,
                'duration_formatted': f"{int(duration//60):02d}:{int(duration%60):02d}"
            }
            
            logger.info(f"Info video: {width}x{height}, {fps:.2f} FPS, {info['duration_formatted']}")
            return info
            
        except Exception as e:
            logger.error(f"Errore lettura info video: {e}")
            return {"error": str(e)}

class MarkdownTranslator:
    """Traduttore specializzato per file Markdown"""
    
    def __init__(self, translator: GPUTranslator):
        self.translator = translator
    
    def translate_markdown_file(self, input_file: str, output_file: Optional[str] = None) -> Optional[str]:
        """Traduce file Markdown preservando formattazione"""
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.error(f"File non trovato: {input_file}")
            return None
        
        if output_file is None:
            output_file = input_path.stem + "_translated.md"
        
        output_path = Path(output_file)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Traduce preservando la struttura Markdown
            translated_content = self._translate_markdown_content(content)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            
            logger.info(f"File tradotto salvato: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Errore traduzione file: {e}")
            return None
    
    def _translate_markdown_content(self, content: str) -> str:
        """Traduce contenuto Markdown preservando sintassi"""
        lines = content.split('\n')
        translated_lines = []
        
        code_block = False
        code_delimiter = None
        
        for line in lines:
            # Rileva inizio/fine blocchi di codice
            if line.strip().startswith('```') or line.strip().startswith('~~~'):
                code_block = not code_block
                if code_block:
                    code_delimiter = line.strip()[:3]
                translated_lines.append(line)
                continue
            
            # Non tradurre il contenuto dei blocchi di codice
            if code_block:
                translated_lines.append(line)
                continue
            
            # Non tradurre righe vuote
            if not line.strip():
                translated_lines.append(line)
                continue
            
            # Traduce diversi tipi di contenuto
            try:
                if self._is_header(line):
                    translated_line = self._translate_header(line)
                elif self._is_link_line(line):
                    translated_line = self._translate_link_line(line)
                elif self._is_image(line):
                    translated_line = line  # Non tradurre tag immagine
                elif self._is_list_item(line):
                    translated_line = self._translate_list_item(line)
                elif self._is_quote(line):
                    translated_line = self._translate_quote(line)
                else:
                    # Traduce contenuto normale
                    translated_line = self.translator.translate_text(line.strip())
                    if not translated_line.strip():
                        translated_line = line
            except Exception as e:
                logger.error(f"Errore traduzione linea '{line[:50]}...': {e}")
                translated_line = line
            
            translated_lines.append(translated_line)
        
        return '\n'.join(translated_lines)
    
    def _is_header(self, line: str) -> bool:
        return line.strip().startswith('#') and ' ' in line.strip()
    
    def _is_link_line(self, line: str) -> bool:
        return '[' in line and '](' in line and ')' in line
    
    def _is_image(self, line: str) -> bool:
        return line.strip().startswith('![')
    
    def _is_list_item(self, line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith('- ') or stripped.startswith('* ') or re.match(r'^\d+\.\s', stripped)
    
    def _is_quote(self, line: str) -> bool:
        return line.strip().startswith('>')
    
    def _translate_header(self, line: str) -> str:
        """Traduce header mantenendo i simboli #"""
        match = re.match(r'^(\s*#+\s*)(.*)', line)
        if match:
            prefix, text = match.groups()
            if text.strip():
                translated_text = self.translator.translate_text(text.strip())
                return f"{prefix}{translated_text}"
        return line
    
    def _translate_link_line(self, line: str) -> str:
        """Traduce testo dei link mantenendo URL"""
        def replace_link(match):
            text, url = match.groups()
            # Non tradurre se il testo è solo un URL
            if text.strip().startswith(('http://', 'https://', 'www.')):
                return f"[{text}]({url})"
            translated_text = self.translator.translate_text(text)
            return f"[{translated_text}]({url})"
        
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, line)
    
    def _translate_list_item(self, line: str) -> str:
        """Traduce elementi lista mantenendo formattazione"""
        match = re.match(r'^(\s*[-*]|\s*\d+\.)\s*(.*)', line)
        if match:
            prefix, text = match.groups()
            if text.strip():
                translated_text = self.translator.translate_text(text.strip())
                return f"{prefix} {translated_text}"
        return line
    
    def _translate_quote(self, line: str) -> str:
        """Traduce citazioni mantenendo >"""
        match = re.match(r'^(\s*>\s*)(.*)', line)
        if match:
            prefix, text = match.groups()
            if text.strip():
                translated_text = self.translator.translate_text(text.strip())
                return f"{prefix}{translated_text}"
        return line

class NotesEnhancer:
    """Classe per migliorare appunti con contenuto video"""
    
    def __init__(self, translator: GPUTranslator, ocr_processor: OCRProcessor):
        self.translator = translator
        self.ocr_processor = ocr_processor
    
    def enhance_notes_with_video(self, notes_file: str, video_frames: List[str], 
                                output_file: Optional[str] = None) -> Optional[str]:
        """Arricchisce appunti con contenuto OCR dai frame del video"""
        
        notes_path = Path(notes_file)
        if not notes_path.exists():
            logger.error(f"File appunti non trovato: {notes_file}")
            return None
        
        if output_file is None:
            output_file = notes_path.stem + "_enhanced_italian.md"
        
        output_path = Path(output_file)
        
        try:
            # Leggi appunti originali
            with open(notes_path, 'r', encoding='utf-8') as f:
                original_notes = f.read()
            
            # Traduci appunti base
            logger.info("Traduzione appunti base...")
            md_translator = MarkdownTranslator(self.translator)
            translated_notes = md_translator._translate_markdown_content(original_notes)
            
            # Elabora frame video per contenuto aggiuntivo
            additional_content = []
            
            if video_frames:
                logger.info(f"Elaborazione {len(video_frames)} frame per contenuto aggiuntivo...")
                
                for i, frame_path in enumerate(video_frames):
                    logger.info(f"Elaborando frame {i+1}/{len(video_frames)}: {Path(frame_path).name}")
                    
                    # Estrai testo da frame
                    ocr_results = self.ocr_processor.extract_text_from_image(frame_path)
                    
                    frame_content = []
                    for engine, text in ocr_results.items():
                        if text and len(text.strip()) > 10:  # Solo se c'è contenuto significativo
                            # Traduci contenuto OCR
                            translated_text = self.translator.translate_text(text)
                            frame_content.append({
                                'engine': engine,
                                'original': text.strip(),
                                'translated': translated_text.strip()
                            })
                    
                    if frame_content:
                        # Estrai timestamp dal nome del file
                        timestamp = self._extract_timestamp_from_filename(frame_path)
                        additional_content.append({
                            'frame': frame_path,
                            'timestamp': timestamp,
                            'content': frame_content
                        })
            
            # Componi documento finale
            enhanced_content = self._compose_enhanced_document(
                translated_notes, additional_content, notes_path.stem
            )
            
            # Salva risultato
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
            
            logger.info(f"Appunti arricchiti salvati: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Errore arricchimento appunti: {e}")
            return None
    
    def _extract_timestamp_from_filename(self, filename: str) -> str:
        """Estrae timestamp dal nome del file frame"""
        # Cerca pattern come "15m30s" nel nome del file
        match = re.search(r'(\d+)m(\d+)s', filename)
        if match:
            minutes, seconds = match.groups()
            return f"{minutes}:{seconds.zfill(2)}"
        return "00:00"
    
    def _compose_enhanced_document(self, base_notes: str, additional_content: List[Dict], 
                                  original_title: str) -> str:
        """Compone documento finale con appunti base + contenuto video"""
        
        content = []
        
        # Header
        content.append(f"# {original_title} - Appunti Tradotti e Arricchiti")
        content.append(f"\n*Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')}*\n")
        
        # Appunti base tradotti
        content.append("## Appunti Principali\n")
        content.append(base_notes)
        content.append("\n")
        
        # Contenuto aggiuntivo da video
        if additional_content:
            content.append("## Contenuto Aggiuntivo dal Video\n")
            content.append("*Testo estratto automaticamente dai frame del video tramite OCR*\n")
            
            for item in additional_content:
                timestamp = item['timestamp']
                frame_name = Path(item['frame']).name
                
                content.append(f"### Timestamp {timestamp}")
                content.append(f"*Frame: {frame_name}*\n")
                
                for ocr_result in item['content']:
                    engine = ocr_result['engine'].upper()
                    original = ocr_result['original']
                    translated = ocr_result['translated']
                    
                    content.append(f"**Testo rilevato ({engine}):**")
                    content.append(f"> {translated}")
                    content.append("")
                    
                    # Aggiungi originale in dettagli nascosti
                    content.append("<details>")
                    content.append("<summary>Testo originale (inglese)</summary>")
                    content.append("")
                    content.append("```")
                    content.append(original)
                    content.append("```")
                    content.append("</details>")
                    content.append("")
                
                content.append("---\n")
        
        # Footer
        content.append("## Informazioni Tecniche")
        content.append("- **Traduzione**: Helsinki-NLP OPUS-MT (GPU accelerata)")
        content.append("- **OCR**: EasyOCR + Tesseract")
        content.append("- **Elaborazione**: Script automatico Python")
        
        return '\n'.join(content)

class ScreenshotManager:
    """Gestisce screenshot automatici e manuali"""
    
    def __init__(self, output_dir: str = "screenshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if HAS_SCREENSHOT:
            # Disabilita fail-safe per automazione
            pyautogui.FAILSAFE = False
        
    def take_screenshot(self, name: Optional[str] = None) -> Optional[str]:
        """Cattura screenshot dello schermo"""
        if not HAS_SCREENSHOT:
            logger.error("Librerie screenshot non disponibili")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png" if name else f"screenshot_{timestamp}.png"
        filepath = self.output_dir / filename
        
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            logger.info(f"Screenshot salvato: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Errore screenshot: {e}")
            return None
    
    def take_region_screenshot(self, region: Tuple[int, int, int, int], 
                             name: Optional[str] = None) -> Optional[str]:
        """Cattura screenshot di una regione specifica (x, y, width, height)"""
        if not HAS_SCREENSHOT:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png" if name else f"region_{timestamp}.png"
        filepath = self.output_dir / filename
        
        try:
            screenshot = pyautogui.screenshot(region=region)
            screenshot.save(filepath)
            logger.info(f"Screenshot regione salvato: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Errore screenshot regione: {e}")
            return None

def check_dependencies() -> Dict[str, bool]:
    """Verifica tutte le dipendenze e restituisce lo stato"""
    deps = {
        "transformers": HAS_TRANSFORMERS,
        "easyocr": HAS_EASYOCR,
        "tesseract": HAS_TESSERACT,
        "opencv": HAS_CV2,
        "screenshot": HAS_SCREENSHOT,
        "cuda": torch.cuda.is_available() if HAS_TRANSFORMERS else False
    }
    return deps

def print_dependency_status():
    """Stampa lo stato delle dipendenze"""
    deps = check_dependencies()
    
    logger.info("=== STATO DIPENDENZE ===")
    for name, available in deps.items():
        status = "OK" if available else "MANCANTE"
        logger.info(f"{name}: {status}")
    
    missing = [name for name, available in deps.items() if not available]
    if missing:
        logger.warning("Per installare dipendenze mancanti:")
        if "transformers" in missing:
            logger.warning("pip install transformers torch")
        if "easyocr" in missing:
            logger.warning("pip install easyocr")
        if "tesseract" in missing:
            logger.warning("pip install pytesseract pillow")
        if "opencv" in missing:
            logger.warning("pip install opencv-python")
        if "screenshot" in missing:
            logger.warning("pip install pyautogui")

def main():
    parser = argparse.ArgumentParser(
        description="Traduttore completo MD + Video + OCR con GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:

  # Traduzione semplice appunti
  python script.py -i appunti.md

  # Traduzione + arricchimento con video (RACCOMANDATO)
  python script.py -i appunti.md -v video.mp4 --enhance-notes

  # Frame ogni 30 secondi invece di 60
  python script.py -i appunti.md -v video.mp4 --enhance-notes --frame-interval 30

  # Usa frame chiave automatici
  python script.py -i appunti.md -v video.mp4 --enhance-notes --key-frames

  # Solo traduzione + OCR sui frame (senza arricchimento)
  python script.py -i appunti.md -v video.mp4

  # Screenshot manuale + OCR
  python script.py -i appunti.md --screenshot

  # OCR su immagine specifica
  python script.py -i appunti.md --ocr immagine.png
        """
    )
    
    # Argomenti principali
    parser.add_argument("--input", "-i", required=True, 
                       help="File Markdown da tradurre")
    parser.add_argument("--video", "-vid", 
                       help="File MP4 per estrazione frame")
    parser.add_argument("--output", "-o", 
                       help="File output (opzionale)")
    
    # Modalità video
    parser.add_argument("--enhance-notes", action="store_true",
                       help="Arricchisci appunti con contenuto video (RACCOMANDATO)")
    parser.add_argument("--frame-interval", type=int, default=60,
                       help="Intervallo frame video in secondi (default: 60)")
    parser.add_argument("--key-frames", action="store_true",
                       help="Estrai frame chiave invece di intervalli fissi")
    
    # Screenshot e OCR
    parser.add_argument("--screenshot", "-s", action="store_true", 
                       help="Cattura screenshot")
    parser.add_argument("--region", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'),
                       help="Screenshot regione (x y width height)")
    parser.add_argument("--ocr", 
                       help="File immagine per OCR")
    
    # Opzioni avanzate
    parser.add_argument("--check-deps", action="store_true",
                       help="Verifica solo le dipendenze")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Output verboso")
    
    args = parser.parse_args()
    
    # Verifica dipendenze se richiesto
    if args.check_deps:
        print_dependency_status()
        return 0
    
    # Configura logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== SCRIPT TRADUZIONE + VIDEO + OCR ===")
    print_dependency_status()
    
    # Verifica dipendenze critiche
    if not HAS_TRANSFORMERS:
        logger.error("ERRORE: transformers non disponibile. Installa con: pip install transformers torch")
        return 1
    
    if args.video and not HAS_CV2:
        logger.error("ERRORE: opencv-python necessario per video. Installa con: pip install opencv-python")
        return 1
    
    if not (HAS_EASYOCR or HAS_TESSERACT):
        logger.warning("ATTENZIONE: Nessun motore OCR disponibile")
    
    # Inizializza componenti
    logger.info("Inizializzazione componenti...")
    try:
        translator = GPUTranslator()
        md_translator = MarkdownTranslator(translator)
        ocr_processor = OCRProcessor()
        screenshot_manager = ScreenshotManager()
        
        # Processamento video se fornito
        video_frames = []
        if args.video:
            logger.info(f"Processando video: {args.video}")
            video_processor = VideoProcessor()
            
            # Ottieni info video
            video_info = video_processor.get_video_info(args.video)
            if "error" not in video_info:
                duration = video_info.get('duration_formatted', 'N/A')
                logger.info(f"Durata video: {duration}")
            
            # Estrai frame
            if args.key_frames:
                logger.info("Estraendo frame chiave...")
                video_frames = video_processor.extract_key_frames(args.video)
            else:
                logger.info(f"Estraendo frame ogni {args.frame_interval} secondi...")
                video_frames = video_processor.extract_frames_from_video(args.video, args.frame_interval)
            
            if video_frames:
                logger.info(f"Estratti {len(video_frames)} frame")
            else:
                logger.warning("Nessun frame estratto dal video")
        
        # Traduzione e arricchimento appunti
        if args.enhance_notes and video_frames:
            logger.info("Arricchimento appunti con contenuto video...")
            notes_enhancer = NotesEnhancer(translator, ocr_processor)
            
            enhanced_file = notes_enhancer.enhance_notes_with_video(
                args.input, video_frames, args.output
            )
            
            if enhanced_file:
                logger.info(f"Appunti arricchiti completati: {enhanced_file}")
            else:
                logger.error("Errore nell'arricchimento appunti")
                return 1
        
        elif args.enhance_notes and not video_frames:
            logger.warning("--enhance-notes richiesto ma nessun frame video disponibile")
            logger.info("Procedo con traduzione semplice...")
            
            translated_file = md_translator.translate_markdown_file(args.input, args.output)
            if translated_file:
                logger.info(f"Traduzione completata: {translated_file}")
            else:
                logger.error("Errore nella traduzione")
                return 1
        
        else:
            # Traduzione semplice
            logger.info("Traduzione file Markdown...")
            translated_file = md_translator.translate_markdown_file(args.input, args.output)
            
            if translated_file:
                logger.info(f"Traduzione completata: {translated_file}")
            else:
                logger.error("Errore nella traduzione")
                return 1
            
            # OCR su frame video se disponibili (senza arricchimento)
            if video_frames:
                logger.info("Elaborazione OCR sui frame video...")
                max_frames = min(len(video_frames), 10)  # Limita per evitare troppi file
                
                for i, frame_path in enumerate(video_frames[:max_frames]):
                    logger.info(f"OCR frame {i+1}/{max_frames}: {Path(frame_path).name}")
                    
                    ocr_results = ocr_processor.extract_text_from_image(frame_path)
                    
                    for engine, text in ocr_results.items():
                        if text and len(text.strip()) > 10:
                            translated_ocr = translator.translate_text(text)
                            
                            # Salva risultato OCR
                            frame_name = Path(frame_path).stem
                            ocr_file = f"{frame_name}_ocr_{engine}_translated.txt"
                            
                            try:
                                with open(ocr_file, 'w', encoding='utf-8') as f:
                                    f.write(f"=== FRAME: {frame_name} ===\n")
                                    f.write(f"=== ENGINE: {engine.upper()} ===\n\n")
                                    f.write(f"ORIGINALE:\n{text}\n\n")
                                    f.write(f"TRADUZIONE:\n{translated_ocr}")
                                
                                logger.info(f"OCR salvato: {ocr_file}")
                            except Exception as e:
                                logger.error(f"Errore salvataggio OCR {ocr_file}: {e}")
        
        # Screenshot se richiesto
        if args.screenshot:
            logger.info("Cattura screenshot...")
            if args.region:
                screenshot_path = screenshot_manager.take_region_screenshot(tuple(args.region), "manual")
            else:
                screenshot_path = screenshot_manager.take_screenshot("manual")
            
            if screenshot_path:
                logger.info(f"Screenshot salvato: {screenshot_path}")
                
                # OCR automatico su screenshot
                logger.info("Elaborazione OCR su screenshot...")
                ocr_results = ocr_processor.extract_text_from_image(screenshot_path)
                
                for engine, text in ocr_results.items():
                    if text:
                        logger.info(f"OCR {engine}: {len(text)} caratteri estratti")
                        
                        # Traduce testo OCR
                        translated_ocr = translator.translate_text(text)
                        
                        # Salva risultati OCR
                        ocr_file = Path(screenshot_path).stem + f"_ocr_{engine}.txt"
                        try:
                            with open(ocr_file, 'w', encoding='utf-8') as f:
                                f.write(f"=== TESTO ORIGINALE ({engine.upper()}) ===\n\n")
                                f.write(text)
                                f.write(f"\n\n=== TRADUZIONE ITALIANA ===\n\n")
                                f.write(translated_ocr)
                            
                            logger.info(f"OCR salvato: {ocr_file}")
                        except Exception as e:
                            logger.error(f"Errore salvataggio OCR: {e}")
            else:
                logger.error("Errore nella cattura screenshot")
        
        # OCR su file immagine specifico
        if args.ocr:
            logger.info(f"Elaborazione OCR su {args.ocr}...")
            ocr_results = ocr_processor.extract_text_from_image(args.ocr)
            
            if not ocr_results:
                logger.warning(f"Nessun testo estratto da {args.ocr}")
            
            for engine, text in ocr_results.items():
                if text:
                    logger.info(f"OCR {engine}: {len(text)} caratteri estratti")
                    translated_ocr = translator.translate_text(text)
                    
                    ocr_file = Path(args.ocr).stem + f"_ocr_{engine}_translated.txt"
                    try:
                        with open(ocr_file, 'w', encoding='utf-8') as f:
                            f.write(f"=== ORIGINALE ({engine.upper()}) ===\n\n{text}\n\n")
                            f.write(f"=== TRADUZIONE ===\n\n{translated_ocr}")
                        
                        logger.info(f"OCR tradotto salvato: {ocr_file}")
                    except Exception as e:
                        logger.error(f"Errore salvataggio OCR: {e}")
        
        logger.info("Elaborazione completata con successo!")
        return 0
        
    except Exception as e:
        logger.error(f"Errore critico durante l'esecuzione: {e}")
        return 1

if __name__ == "__main__":
    exit(main())