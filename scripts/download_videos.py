import pandas as pd
import os
import re
from pathlib import Path
import io # Import for handling file streams

# Google API libraries
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload # New import for downloading
from tqdm import tqdm # For progress bars

# --- SCRIPT CONFIGURATION ---

# 1. PASTE YOUR GOOGLE DRIVE FOLDER ID HERE
GDRIVE_FOLDER_ID = "1pxJap9MnQ8W18k2YVx8c26BU220b4K7Z" 

# 2. Define project paths
CSV_FILE_PATH = 'dataset/poses_master_list.csv'
DOWNLOAD_DIR = 'videos/'
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json' 

# 3. Define the "scopes" (permissions) we need
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# --- END CONFIGURATION ---

def authenticate_gdrive():
    """Handles Google Drive authentication and returns a service object."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"ERROR: '{CREDENTIALS_FILE}' not found.")
                print("Please follow the setup steps to download it from Google Cloud.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        print(f"An error occurred building the service: {e}")
        return None

def clean_filename(filename):
    """Converts 'warrior_II_1.mp4' to 'WarriorII_1'."""
    name_without_ext = os.path.splitext(filename)[0]
    name_capitalized = re.sub(r"[ _-]([a-zA-Z])", 
                              lambda m: m.group(1).upper(), 
                              name_without_ext)
    return name_capitalized[0].upper() + name_capitalized[1:]

def fetch_files_and_create_csv(service):
    """Fetches files from GDrive, creates the CSV, and returns a DataFrame."""
    if GDRIVE_FOLDER_ID == "PASTE_YOUR_FOLDER_ID_HERE":
        print("ERROR: Please paste your GDRIVE_FOLDER_ID at the top of the script.")
        return None

    print(f"Searching for files in Google Drive Folder ID: {GDRIVE_FOLDER_ID}")
    
    query = f"'{GDRIVE_FOLDER_ID}' in parents and mimeType contains 'video/'"
    
    try:
        results = service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name)" # Simplified fields
        ).execute()
        
        files = results.get('files', [])
    except Exception as e:
        print(f"An error occurred while querying the Drive API: {e}")
        print("Please check your FOLDER_ID and API permissions.")
        return None

    if not files:
        print("No video files found in that Google Drive folder.")
        return None

    print(f"Found {len(files)} video files. Generating master list...")
    
    pose_data = []
    for i, file in enumerate(files):
        pose_id = i
        video_filename = file.get('name')
        pose_name = clean_filename(video_filename)
        gdrive_id = file.get('id') # We just need the ID
        
        pose_data.append({
            'pose_id': pose_id,
            'pose_name': pose_name,
            'video_filename': video_filename,
            'gdrive_id': gdrive_id # Store the ID directly
        })

    df = pd.DataFrame(pose_data)
    
    os.makedirs('dataset', exist_ok=True)
    df.to_csv(CSV_FILE_PATH, index=False)
    
    print(f"Successfully auto-generated '{CSV_FILE_PATH}'!")
    return df

# --- MODIFIED FUNCTION ---
def download_videos_from_df(df, service):
    """
    Downloads all videos listed in the DataFrame using the authenticated
    Google Drive API service.
    """
    if df is None or df.empty:
        print("No data in DataFrame, skipping download.")
        return
        
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    total_files = len(df)
    
    print(f"\n--- Starting Download Process ({total_files} files) ---")
    
    # Use tqdm for a single progress bar
    for index, row in tqdm(df.iterrows(), total=total_files, desc="Downloading Videos"):
        video_name = row['video_filename']
        file_id = row['gdrive_id'] # Get the ID from the DataFrame
        output_path = os.path.join(DOWNLOAD_DIR, video_name)
        
        if os.path.exists(output_path):
            # print(f"[{index + 1}/{total_files}] SKIPPING: '{video_name}' already exists.")
            continue
            
        try:
            # Create a request to get the file media
            request = service.files().get_media(fileId=file_id)
            
            # Prepare a file handle to write the video
            with open(output_path, 'wb') as fh:
                # Use MediaIoBaseDownload to download in chunks
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    # Optional: Add progress bar for individual file
                    # print(f"Downloading {video_name}: {int(status.progress() * 100)}%")

            # print(f"Successfully downloaded '{video_name}'.")
        except Exception as e:
            print(f"ERROR: Failed to download '{video_name}'. Reason: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
    
    print("\n--- Download process complete! ---")
    print(f"All videos are in the '{DOWNLOAD_DIR}' folder.")

if __name__ == "__main__":
    service = authenticate_gdrive()
    if service:
        # 1. Create the CSV
        master_list_df = fetch_files_and_create_csv(service)
        
        # 2. Download the files using the authenticated service
        if master_list_df is not None:
            # Pass the 'service' object to the download function
            download_videos_from_df(master_list_df, service)
        else:
            print("Failed to create master list. Cannot proceed with download.")