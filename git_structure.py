import os

def build_structure():
    # Define the directory structure
    structure = {
        "data": ["Phishing_Legitimate_full.arff"],
        "src": ["__init__.py", "data_loader.py", "preprocessing.py", "ga_selector.py", "models.py"],
        "notebooks": ["exploration.ipynb"],
        ".": ["main.py", "requirements.txt", ".gitignore", "README.md"]
    }

    for folder, files in structure.items():
        # Create folders (except for the root '.')
        if folder != ".":
            os.makedirs(folder, exist_ok=True)
            print(f"✔ Created folder: {folder}")

        # Create files
        for file in files:
            file_path = os.path.join(folder, file) if folder != "." else file
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Pre-populate .gitignore with standard Python exclusions
                    if file == ".gitignore":
                        f.write("__pycache__/\n*.py[cod]\nvenv/\n.env\n.ipynb_checkpoints/\ndata/*.arff\n")
                    # Pre-populate README
                    elif file == "README.md":
                        f.write("# Genetic Algorithm-Based Phishing Detection\n\nResearch reproduction project.")
                print(f"✔ Created file: {file_path}")
            else:
                print(f"ℹ File already exists: {file_path}")

if __name__ == "__main__":
    build_structure()
    print("\n🚀 Project structure built successfully!")