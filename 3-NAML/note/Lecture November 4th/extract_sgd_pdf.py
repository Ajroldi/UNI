import pdfplumber
import sys

pdf_path = r'C:\Users\miche\OneDrive\Desktop\UNI\3-NAML\note\Lecture November 4th\SGD_v1.pdf'
output_path = r'C:\Users\miche\OneDrive\Desktop\UNI\3-NAML\note\Lecture November 4th\sgd_slides.txt'

try:
    with pdfplumber.open(pdf_path) as pdf:
        total_slides = len(pdf.pages)
        print(f"SGD_v1.pdf: {total_slides} slides")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"=== SGD_v1.pdf - {total_slides} Slides ===\n\n")
            
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                images = page.images
                
                f.write(f"--- Slide {i}/{total_slides} ---\n")
                f.write(f"Images: {len(images)}\n\n")
                
                if text:
                    f.write(text)
                else:
                    f.write("[Slide without extractable text - likely image/diagram only]")
                
                f.write("\n\n" + "="*80 + "\n\n")
        
        print(f"✓ Extracted to: {output_path}")
        
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
