import os
import sys

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[CREATED] Directory: {path}")
    else:
        print(f"[SKIPPED] Directory already exists: {path}")

def handle_gitignore():
    gitignore_path = ".gitignore"
    project_specific_entries = ["data/", "*.sxm", "*.dat"]
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            existing = f.read()
        existing_lines = [line.strip() for line in existing.splitlines() if line.strip()]
        
        new_entries = []
        for entry in project_specific_entries:
            if entry not in existing_lines:
                new_entries.append(entry)
        
        if new_entries:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_entries) + "\n")
            print(f"[APPENDED] .gitignore with new entries: {new_entries}")
        else:
            print("[SKIPPED] .gitignore already contains all project-specific entries")
    else:
        standard_entries = [
            "# Python",
            "__pycache__/",
            "*.py[cod]",
            "*$py.class",
            "*.so",
            ".Python",
            "env/",
            "venv/",
            ".env",
            "*.egg-info/",
            "*.egg",
            "dist/",
            "build/",
            ".tox/",
            ".mypy_cache/",
            ".pytest_cache/",
            "# IDE",
            ".vscode/",
            ".idea/",
            "# Project specific (physics/data)",
            "data/",
            "*.sxm",
            "*.dat",
        ]
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("\n".join(standard_entries) + "\n")
        print(f"[CREATED] .gitignore with standard and project-specific entries")

def create_readme():
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        content = """# MBE_Condition_prediction

Minimal setup for MBE condition prediction project.

## Requirements
- Python 3.8+
- See requirements.txt

## License
MIT License (placeholder)
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[CREATED] README.md")
    else:
        print(f"[SKIPPED] README.md already exists")

def create_requirements():
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        packages = [
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "torch",
        ]
        with open(req_path, "w", encoding="utf-8") as f:
            f.write("\n".join(packages) + "\n")
        print(f"[CREATED] requirements.txt with: {packages}")
    else:
        print(f"[SKIPPED] requirements.txt already exists")

def main():
    base_dir = os.getcwd()
    print(f"[INFO] Working directory: {base_dir}")
    
    # Create directories
    dirs = ["test", "src", "notebooks"]
    for d in dirs:
        create_directory(d)
    
    # Handle .gitignore
    handle_gitignore()
    
    # Create README.md
    create_readme()
    
    # Create requirements.txt
    create_requirements()
    
    print("\n[DONE] Setup complete.")

if __name__ == "__main__":
    main()