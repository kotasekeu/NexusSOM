# # organize_results.py
# import os
# import shutil
# from tqdm import tqdm
#
#
# def organize_ea_results(base_dir="."):
#     """
#     Organizes visualization files from individual UID folders into typed subdirectories.
#
#     This script searches for an 'individuals' directory in the base_dir,
#     iterates through each UID subdirectory, and copies relevant plot files
#     to newly created 'mqe_plots', 'distance_maps', and 'u_matrix_maps' folders.
#     Files are renamed using their parent UID to ensure uniqueness.
#     """
#     individuals_path = os.path.join(base_dir, "individuals")
#
#     if not os.path.isdir(individuals_path):
#         print(f"Error: Directory '{individuals_path}' not found.")
#         print("Please run this script from the root directory of your EA results (e.g., 'ea-iris-20x30/').")
#         return
#
#     # 1. Create target directories
#     output_dirs = {
#         'mqe': os.path.join(base_dir, "mqe_plots"),
#         'distance': os.path.join(base_dir, "distance_maps"),
#         'umatrix': os.path.join(base_dir, "u_matrix_maps")
#     }
#
#     for dir_path in output_dirs.values():
#         os.makedirs(dir_path, exist_ok=True)
#
#     print("Created output directories: mqe_plots, distance_maps, u_matrix_maps")
#
#     # 2. Get list of all UID directories to process
#     uid_folders = [d for d in os.listdir(individuals_path) if os.path.isdir(os.path.join(individuals_path, d))]
#
#     if not uid_folders:
#         print("No individual UID folders found in 'individuals' directory.")
#         return
#
#     print(f"Found {len(uid_folders)} individual result folders to process.")
#
#     # 3. Iterate, copy, and rename files
#     copied_files_count = 0
#     for uid in tqdm(uid_folders, desc="Organizing files"):
#         uid_folder_path = os.path.join(individuals_path, uid)
#
#         # Source file paths
#         source_files = {
#             'mqe': os.path.join(uid_folder_path, "visualizations", "mqe_evolution.png"),
#             'distance': os.path.join(uid_folder_path, "visualizations", "distance_map.png"),
#             'umatrix': os.path.join(uid_folder_path, "visualizations", "u_matrix.png")
#         }
#
#         # Destination file paths with new names
#         dest_files = {
#             'mqe': os.path.join(output_dirs['mqe'], f"{uid}_mqe.png"),
#             'distance': os.path.join(output_dirs['distance'], f"{uid}_distance.png"),
#             'umatrix': os.path.join(output_dirs['umatrix'], f"{uid}_umatrix.png")
#         }
#
#         # Copy files if they exist
#         for key in source_files:
#             if os.path.exists(source_files[key]):
#                 try:
#                     shutil.copy2(source_files[key], dest_files[key])  # copy2 preserves metadata
#                     copied_files_count += 1
#                 except Exception as e:
#                     print(f"\nWarning: Could not copy file for UID {uid}. Error: {e}")
#
#     print(f"\nOrganization complete. Copied {copied_files_count} files successfully.")
#
#
# if __name__ == "__main__":
#     # The script will run in the directory where it is located.
#     organize_ea_results()