from pathlib import Path

# Set your root directory
root = Path("fruit_disease_dataset")
valid_suffixes = [".jpg", ".jpeg", ".png"]
fix_count = 0

print(f"🔍 Scanning and fixing double file extensions in: {root.resolve()}\n")

for f in root.rglob("*"):
    if not f.is_file():
        continue
    name = f.name.lower()
    for sfx in valid_suffixes:
        if name.endswith(sfx + sfx):
            new_name = f.name.replace(sfx + sfx, sfx)
            new_path = f.with_name(new_name)
            f.rename(new_path)
            print(f"🛠️ Fixed: {f.name} → {new_name}")
            fix_count += 1

print(f"\n✅ Fix complete. Total corrected files: {fix_count}")
