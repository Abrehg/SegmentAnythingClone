import os
import shutil

def group_files_by_name(base_folder):
    def move_files_to_target(base_folder):
        for root, _, files in os.walk(base_folder):
            # Create a list of directories in the current level
            existing_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                file_path = os.path.join(root, file)
                
                # Check if there's a corresponding directory for this base name
                for dir_name in existing_dirs:
                    if dir_name in file_name:
                        target_folder = os.path.join(root, dir_name)
                        shutil.move(file_path, os.path.join(target_folder, os.path.basename(file_path)))
                        break
    
    move_files_to_target(base_folder)

    # Traverse other folders in the base folder
    for root, dirs, _ in os.walk(base_folder):
        for dir in dirs:
            group_files_by_name(os.path.join(root, dir))
            return

# Base folder path
base_folder = '/Users/adityaasuratkal/Downloads/Img_Data/ADE20K/ADE20K_2021_17_01/images/ADE'
#bathroom_folder = '/Users/adityaasuratkal/Downloads/Img_Data/ADE20K/ADE20K_2021_17_01/images/ADE/training/home_or_hotel'
group_files_by_name(base_folder)