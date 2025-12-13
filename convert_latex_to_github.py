import re
import os

def convert_latex_to_github(content):
    """
    Converte formule LaTeX display $$...$$ in blocchi ```math``` per GitHub.
    Le formule inline $...$ rimangono invariate (supportate da GitHub).
    """
    # Pattern per trovare blocchi $$...$$ (anche multilinea)
    # Usa DOTALL per matchare anche newline
    pattern = r'\$\$(.*?)\$\$'
    
    def replace_display_math(match):
        formula = match.group(1).strip()
        return f"```math\n{formula}\n```"
    
    # Sostituisce tutti i blocchi $$...$$ con ```math```
    converted = re.sub(pattern, replace_display_math, content, flags=re.DOTALL)
    
    return converted

def process_file(filepath):
    """Processa un singolo file markdown."""
    print(f"Processing: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Converti le formule
        converted_content = convert_latex_to_github(content)
        
        # Salva il file modificato
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(converted_content)
        
        print(f"  ✓ Converted successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    base_path = r"C:\Users\miche\OneDrive\Desktop\UNI\3-NAML"
    
    # Lista dei file da Lez12 a Lez30
    files = [
        "Lez12-13ott.md",
        "Lez13-17ott.md",
        "Lez14-20ott.md",
        "Lez15-21ott.md",
        "Lez16-27ott.md",
        "Lez17-31ott.md",
        "Lez18-3nov.md",
        "Lez19-4nov.md",
        "Lez20-07nov.md",
        "Lez21-10nov.md",
        "Lez22-14nov.md",
        "Lez23-17nov.md",
        "Lez24-21nov.md",
        "Lez25-25nov.md",
        "Lez26-28nov.md",
        "Lez27-02dic.md",
        "Lez28-05dic.md",
        "Lez29-09dic.md",
        "Lez30-12dic.md"
    ]
    
    success_count = 0
    total_count = 0
    
    for filename in files:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            total_count += 1
            if process_file(filepath):
                success_count += 1
        else:
            print(f"File not found: {filepath}")
    
    print(f"\n{'='*50}")
    print(f"Conversion complete: {success_count}/{total_count} files processed")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
