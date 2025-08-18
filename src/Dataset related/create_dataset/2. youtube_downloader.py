# youtube_fetcher.py

import yt_dlp
import os
import pandas as pd
import re

DOWNLOAD_DIR = "./youtube_videos"
COOKIE_PATH = os.path.join(os.path.dirname(__file__), "cookies.txt")


def clean_title(title):
    """Remove colons and trim for filename safety."""
    return re.sub(r'[\\/:*?"<>|]', '', title).strip()


def get_existing_titles():
    """Scan the download folder and extract cleaned titles."""
    existing_titles = set()
    if os.path.exists(DOWNLOAD_DIR):
        for filename in os.listdir(DOWNLOAD_DIR):
            if filename.endswith(".mp4") and "_" in filename:
                parts = filename.split("_", 1)
                if len(parts) == 2:
                    title_part = parts[1].rsplit(".", 1)[0].lower()
                    existing_titles.add(title_part)
    return existing_titles


def get_next_index():
    indices = []
    if os.path.exists(DOWNLOAD_DIR):
        for filename in os.listdir(DOWNLOAD_DIR):
            if filename.endswith(".mp4") and "_" in filename:
                index_part = filename.split("_", 1)[0]
                if index_part.isdigit():
                    indices.append(int(index_part))
    return max(indices) + 1 if indices else 1


def search_and_download(query: str, max_results=3, min_duration_sec=90, max_duration_sec= 180):
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    if not os.path.exists(COOKIE_PATH):
        print("cookies.txt not found. Please place it in the project folder.")
        return []

    print(f"Searching YouTube for: {query}")

    existing_titles = get_existing_titles()
    video_index = get_next_index()

    ydl_opts = {
        'quiet': False,
        'default_search': 'ytsearch10',
        'noplaylist': True,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'cookiefile': COOKIE_PATH,
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title).70s.%(ext)s'),
    }

    downloaded = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(query, download=False)
        except Exception as e:
            print(f"Search failed: {e}")
            return []

        entries = search_results.get('entries', [])

        for entry in entries:
            title = clean_title(entry.get('title', ''))
            duration = entry.get('duration')
            video_id = entry.get('id')

            if not duration or duration < min_duration_sec or duration > max_duration_sec:
                print(f"Skipping (not in duration range): {title} ({duration} sec)")
                continue

            title_key = title.lower()
            if title_key in existing_titles:
                print(f"Skipping (already downloaded): {title}")
                continue

            print(f"Downloading: {title} ({duration} sec)")
            try:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                info = ydl.extract_info(video_url, download=True)
                original_filename = ydl.prepare_filename(info).replace(".webm", ".mp4")

                new_filename = f"{video_index}_{title}.mp4"
                os.rename(original_filename, os.path.join(DOWNLOAD_DIR, new_filename))

                downloaded.append((new_filename, title, video_url))
                video_index += 1

                if len(downloaded) >= max_results:
                    break

            except Exception as e:
                print(f"Error downloading {title}: {e}")

    print(f"\nDownloaded {len(downloaded)} videos.")
    return downloaded
