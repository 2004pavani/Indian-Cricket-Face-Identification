from pathlib import Path
import shutil

# ==============================
# PROJECT PATHS
# ==============================

PROJECT_DIR = Path(r"E:\Indian-Cricket-Face-Identification")

# Kaggle dataset already inside project
KAGGLE_DATASET = PROJECT_DIR / "dataset"

# GitHub dataset still in Downloads
GITHUB_DATASET = Path(
    r"C:\Users\bajip\Downloads\cricket-face-identifier-main"
    r"\cricket-face-identifier-main\dataset"
)

# Temporary combined dataset
COMBINED_DATASET = PROJECT_DIR / "combined_dataset"


# ==============================
# PLAYER NAME MAPPING
# ==============================

PLAYERS = {
    "Virat_Kohli": {
        "kaggle": "Virat_Kohli",
        "github": "virat_kohli"
    },
    "Rohit_Sharma": {
        "kaggle": "Rohit_Sharma",
        "github": "rohit_sharma"
    },
    "MS_Dhoni": {
        "kaggle": "MS_Dhoni",
        "github": "ms_dhoni"
    },
    "Jasprit_Bumrah": {
        "kaggle": "Jasprit_Bumrah",
        "github": "jasprit_bumrah"
    },
    "Hardik_Pandya": {
        "kaggle": "Hardik_Pandya",
        "github": "hardik_pandya"
    }
}


# ==============================
# CREATE COMBINED DATASET
# ==============================

def copy_images(source_folder, destination_folder, source_name):
    """
    Copy images from one player's dataset into the combined dataset.
    """

    if not source_folder.exists():
        print(f"[WARNING] Not found: {source_folder}")
        return 0

    destination_folder.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_file in source_folder.iterdir():

        if image_file.is_file() and image_file.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp"
        }:

            new_name = f"{source_name}_{image_file.name}"

            destination_file = destination_folder / new_name

            shutil.copy2(image_file, destination_file)

            count += 1

    return count


def main():

    print("=" * 60)
    print("CREATING COMBINED CRICKET DATASET")
    print("=" * 60)

    COMBINED_DATASET.mkdir(parents=True, exist_ok=True)

    total = 0

    for player, folders in PLAYERS.items():

        print(f"\nProcessing: {player}")

        destination = COMBINED_DATASET / player

        kaggle_count = copy_images(
            KAGGLE_DATASET / folders["kaggle"],
            destination,
            "kaggle"
        )

        github_count = copy_images(
            GITHUB_DATASET / folders["github"],
            destination,
            "github"
        )

        player_total = kaggle_count + github_count

        total += player_total

        print(f"  Kaggle : {kaggle_count}")
        print(f"  GitHub : {github_count}")
        print(f"  Total  : {player_total}")

    print("\n" + "=" * 60)
    print(f"TOTAL IMAGES COPIED: {total}")
    print(f"Combined dataset: {COMBINED_DATASET}")
    print("=" * 60)


if __name__ == "__main__":
    main()