import h5py
import os
import sys

def merge_h5_files(source_files, output_file):
    """
    Merges multiple HDF5 files into a single file, putting each source 
    file's content into its own group to avoid naming conflicts.
    """
    print(f"--- HDF5 Merge Process Started ---")
    print(f"Output File: {output_file}")
    
    # Check if any files exist before starting
    existing_files = [f for f in source_files if os.path.exists(f)]
    if not existing_files:
        print("Error: None of the source files were found.")
        return False

    try:
        with h5py.File(output_file, 'w') as dest_h5:
            for src_path in source_files:
                if not os.path.exists(src_path):
                    print(f"Warning: Skipping {src_path} (File not found)")
                    continue
                
                print(f"Processing: {src_path}...")
                # Create a group name based on the filename (without .h5)
                group_name = os.path.splitext(os.path.basename(src_path))[0]
                
                with h5py.File(src_path, 'r') as src_h5:
                    # Create a group for this model
                    group = dest_h5.create_group(group_name)
                    
                    # Copy all datasets and groups from the source file into this group
                    for key in src_h5.keys():
                        src_h5.copy(key, group)
                    
                    # Copy root-level attributes
                    for attr_name, attr_value in src_h5.attrs.items():
                        group.attrs[attr_name] = attr_value
                        
                    print(f"  Successfully copied content from {src_path} to group /{group_name}")

        print(f"--- Merge Complete! ---")
        return True
    except Exception as e:
        print(f"An error occurred during merging: {e}")
        return False

def verify_merged_file(file_path):
    """
    Verifies the contents of the merged HDF5 file.
    """
    print(f"\n--- Verifying Merged File: {file_path} ---")
    if not os.path.exists(file_path):
        print("Error: Merged file does not exist.")
        return

    try:
        with h5py.File(file_path, 'r') as h5:
            print(f"File Size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            print("Groups found in merged file:")
            for group_name in h5.keys():
                group = h5[group_name]
                print(f"- /{group_name} ({len(group.keys())} items)")
                # Sample a dataset if available
                if len(group.keys()) > 0:
                    first_key = list(group.keys())[0]
                    if isinstance(group[first_key], h5py.Dataset):
                        print(f"  Example Dataset: {first_key}, Shape: {group[first_key].shape}")
            
            print("\nVerification successful! The file can be opened and read.")
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    # List of files provided by the user
    files = [
        'plant_model.h5',
        'plant_model_fixed.h5', 
        'plant_disease_model.h5',
        'plant_disease_model_20260421_090847.h5',
        'plant_disease_model_20260421_054458.h5',
        'plant_model_backup.h5'
    ]
    
    output_name = 'combined_plant_models.h5'
    
    # Run the merge
    if merge_h5_files(files, output_name):
        # Verify the result
        verify_merged_file(output_name)
    else:
        print("Merge failed. Please check the error messages above.")
