import glob
import os

def merge_all_xml_to_input():
    output_file = "data/xpp/input.txt"
    
    # ==================== MODIFY THIS PATH ====================
    # Change this to your own X++ XML files folder
    # Example:
    #   - "/home/yourname/MyXppProject"
    #   - "/mnt/c/AOSService/PackagesLocalDirectory"
    #   - "/home/fredshen/MyCustomModels"
    xml_folder_path = "/home/fredshen/XppTools"   # ←←← Change this line to your folder
    
    # Search all XML files recursively
    xml_files = glob.glob(f"{xml_folder_path}/**/*.xml", recursive=True)
    
    print(f"Found {len(xml_files)} XML files in '{xml_folder_path}'")
    print("Starting to merge...")
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i, xml_path in enumerate(xml_files):
            try:
                with open(xml_path, "r", encoding="utf-8", errors="ignore") as xmlf:
                    content = xmlf.read().strip()
                    if len(content) > 150:   # Filter out very small or empty files
                        f.write(content)
                        f.write("\n\n<|file_separator|>\n\n")
                        count += 1
                        if count % 100 == 0:
                            print(f"Merged {count} valid files so far...")
            except Exception:
                pass
    
    print(f"\n✅ Merge completed successfully!")
    print(f"Total valid X++ files merged: {count}")
    print(f"input.txt size: {os.path.getsize(output_file)/1024/1024:.2f} MB")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    merge_all_xml_to_input()
