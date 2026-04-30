import glob
import os
import xml.etree.ElementTree as ET

def extract_xpp_code(xml_file: str):
    """
    Extract X++ source code from a Dynamics 365 XML file.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Look for Source tag which contains the actual X++ code
        for elem in root.iter():
            if elem.text and any(tag in elem.tag.lower() for tag in ['source', 'xpp']):
                code = elem.text.strip()
                if len(code) > 100:   # Filter out empty or very short files
                    return code
        return None
    except:
        return None


def prepare_xpp_data(packages_path: str, max_files: int = 5000):
    """
    Extract X++ code from all XML files and save to input.txt
    """
    os.makedirs("data/xpp", exist_ok=True)
    output_file = "data/xpp/input.txt"
    
    # Search for all XML files recursively
    xml_files = glob.glob(f"{packages_path}/**/*.xml", recursive=True)
    print(f"Found {len(xml_files)} XML files. Starting extraction...")
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for xml_path in xml_files:
            if count >= max_files:
                break
                
            code = extract_xpp_code(xml_path)
            if code:
                f.write(code)
                f.write("\n\n<|file_separator|>\n\n")
                count += 1
                
                if count % 300 == 0:
                    print(f"Extracted {count} valid X++ files...")
    
    print(f"\n✅ Extraction completed!")
    print(f"Total valid X++ files extracted: {count}")
    print(f"input.txt size: {os.path.getsize(output_file)/1024/1024:.2f} MB")
    print(f"Output saved to: {output_file}")


# ==================== MODIFY THIS PATH ====================
if __name__ == "__main__":
    # Change this path to your X++ XML files folder
    # Examples:
    #   "/home/yourname/XppTools"
    #   "/mnt/c/AOSService/PackagesLocalDirectory"
    #   "/home/fredshen/MyCustomModels"
    xpp_folder_path = "/home/fredshen/XppTools"     # ←←← Change this line
    
    prepare_xpp_data(xpp_folder_path, max_files=8000)
