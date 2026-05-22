import zipfile
import os


def fix_model_zip(zip_path):
    if not os.path.exists(zip_path):
        print(f"Error: Could not find {zip_path}")
        return

    temp_path = "models/temp_model.zip"

    with zipfile.ZipFile(zip_path, 'r') as zin:
        with zipfile.ZipFile(temp_path, 'w') as zout:
            for item in zin.infolist():
                # Copy everything EXCEPT the optimizer state
                if "optimizer.pth" not in item.filename:
                    zout.writestr(item, zin.read(item.filename))

    # Overwrite the original zip with the cleaned version
    os.replace(temp_path, zip_path)
    print(f"[FIX] Stripped optimizer from {zip_path}.")
    print("The Utility Agent will now load perfectly for inference!")


if __name__ == "__main__":
    fix_model_zip("models/utility_gnn_ppo.zip")