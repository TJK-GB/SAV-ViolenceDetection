# youtube_fetcher1.py

import yt_dlp
import os
import re
import pandas as pd

DOWNLOAD_DIR = "./youtube_videos"
COOKIE_PATH = os.path.join(os.path.dirname(__file__), "cookies.txt") # cookies are for YouTube for age verification.
ANNOTATION_PATH = "annotations.xlsx"

def clean_filename(title):
    return re.sub(r'[\\/:*?"<>|]', '', title)

def get_existing_titles():
    if not os.path.exists(ANNOTATION_PATH):
        return set()
    try:
        df = pd.read_excel(ANNOTATION_PATH)
        return set(df['Title'].astype(str).str.strip())
    except:
        return set()

def get_next_number():
    existing_files = os.listdir(DOWNLOAD_DIR)
    numbers = []
    for filename in existing_files:
        match = re.match(r'^(\d+)_', filename)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers + [0]) + 1

def search_and_download(query: str, max_results=5, min_duration_sec=90):
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    if not os.path.exists(COOKIE_PATH):
        print("cookies.txt not found. Please export it and place it in the project folder.")
        return []

    existing_titles = get_existing_titles()
    print(f"Searching YouTube for: {query}")

    ydl_opts = {
        'quiet': True,
        'default_search': 'ytsearch10',
        'noplaylist': True,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'cookiefile': COOKIE_PATH,
    }

    downloaded = []
    current_number = get_next_number()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(query, download=False)
        except Exception as e:
            print(f"Search failed: {e}")
            return []

        entries = search_results.get('entries', [])

        for entry in entries:
            title = entry.get('title', '').strip()
            duration = entry.get('duration')
            video_id = entry.get('id')

            if not duration or duration < min_duration_sec:
                continue

            if title in existing_titles:
                continue

            print(f"Downloading: {title}")

            try:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                info = ydl.extract_info(video_url, download=True)
                original_path = ydl.prepare_filename(info).replace(".webm", ".mp4")

                safe_title = clean_filename(title)
                new_filename = f"{current_number}_{safe_title}.mp4"
                new_path = os.path.join(DOWNLOAD_DIR, new_filename)

                if os.path.exists(original_path):
                    os.rename(original_path, new_path)
                    downloaded.append((new_path, title, video_url))
                    current_number += 1

                if len(downloaded) >= max_results:
                    break

            except Exception as e:
                print(f"Error downloading {title}: {e}")

    print(f"Downloaded {len(downloaded)} videos.")
    return downloaded
