

import os
import subprocess
from flask import Flask, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import speech_recognition as sr
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import yt_dlp as youtube_dl
import time
from sklearn.metrics import precision_recall_fscore_support
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_from_youtube(url, path):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(path, 'youtube_audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        audio_file_path = os.path.join(path, 'youtube_audio.wav')
        return audio_file_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def transcribe_audio_chunks(audio_path, recognizer):
    audio = AudioSegment.from_wav(audio_path)
    chunk_length_ms = 60000  # 1-minute chunks
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    full_text = ""

    for i, chunk in enumerate(chunks):
        chunk_filename = os.path.join(app.config['PROCESSED_FOLDER'], f"chunk_{i}.wav")
        chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            full_text += text + " "
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")

        os.remove(chunk_filename)  # Clean up the chunk file

    return full_text.strip()

def generate_keywords(video_content, method):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
    elif method == 'count':
        vectorizer = CountVectorizer(stop_words='english')
    else:
        raise ValueError("Invalid method. Choose 'tfidf' or 'count'.")

    start_time = time.time()
    X = vectorizer.fit_transform([video_content])
    end_time = time.time()
    keyword_generation_time = end_time - start_time
    keywords = vectorizer.get_feature_names_out()
    return keywords, keyword_generation_time

def extract_keywords_from_text_file(input_filename, method):
    text_file_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{input_filename}.txt")
    if not os.path.isfile(text_file_path):
        raise FileNotFoundError(f"Text file '{text_file_path}' not found.")
    
    with open(text_file_path, 'r') as file:
        text = file.read()
    
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
    elif method == 'count':
        vectorizer = CountVectorizer(stop_words='english')
    else:
        raise ValueError("Invalid method. Choose 'tfidf' or 'count'.")

    start_time = time.time()
    X = vectorizer.fit_transform([text])
    end_time = time.time()
    keyword_extraction_time = end_time - start_time
    keywords = vectorizer.get_feature_names_out()
    return keywords, keyword_extraction_time

def calculate_keyword_accuracy(generated_keywords, extracted_keywords):
    intersection = len(set(generated_keywords) & set(extracted_keywords))
    union = len(set(generated_keywords) | set(extracted_keywords))
    precision = intersection / len(generated_keywords)
    recall = intersection / len(extracted_keywords)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1_score

def punctuate_text(text):
    punctuator_url = "http://bark.phon.ioc.ee/punctuator"
    response = requests.post(punctuator_url, data={'text': text})
    if response.status_code == 200:
        return response.text.strip()
    else:
        print(f"Failed to punctuate text: {response.status_code}")
        return text  # Return original text if punctuator fails

def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)
        return text_transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_transcript_to_file(transcript, filename):
    with open(filename, 'w') as file:
        file.write(transcript)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    video_file = request.files.get('video')
    youtube_url = request.form.get('youtube_url')
    method = request.form.get('method', 'tfidf')

    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)

        audio_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{filename.rsplit('.', 1)[0]}.wav")
        subprocess.run(['ffmpeg', '-i', filepath, audio_path])

    elif youtube_url:
        if not youtube_url.startswith(('http://', 'https://')):
            return "Invalid YouTube URL", 400
        video_id = youtube_url.split('v=')[-1]
        transcript = get_youtube_transcript(video_id)
        if transcript:
            filename = f"{video_id}.txt"
            save_transcript_to_file(transcript, os.path.join(app.config['PROCESSED_FOLDER'], filename))
            youtube_transcript = transcript
        else:
            return "Failed to get YouTube transcript", 400
        audio_path = extract_audio_from_youtube(youtube_url, app.config['PROCESSED_FOLDER'])
        if not audio_path:
            return "Failed to process YouTube video", 400
    else:
        return "No file or URL provided", 400

    recognizer = sr.Recognizer()
    full_text = transcribe_audio_chunks(audio_path, recognizer)

    expected_keywords, keyword_generation_time = generate_keywords(full_text, method)

    input_filename = filename.rsplit('.', 1)[0]
    text_filename = f"{input_filename}.txt"
    text_path = os.path.join(app.config['PROCESSED_FOLDER'], text_filename)
    with open(text_path, 'w') as text_file:
        text_file.write(full_text)

    try:
        extracted_keywords, keyword_extraction_time = extract_keywords_from_text_file(input_filename, method)
    except FileNotFoundError as e:
        return str(e), 400

    keywords_generated_list = expected_keywords.tolist()

    if expected_keywords.any():
        intersection = len(set(keywords_generated_list) & set(extracted_keywords))
        coverage_score = intersection / len(expected_keywords)
    else:
        coverage_score = 0.0

    precision, recall, f1_score = calculate_keyword_accuracy(keywords_generated_list, extracted_keywords)

    if youtube_url:
        youtube_keywords, youtube_keyword_generation_time = generate_keywords(youtube_transcript, method)
        youtube_keywords_list = youtube_keywords.tolist()
        precision, recall, f1_score = calculate_keyword_accuracy(keywords_generated_list, youtube_keywords_list)
    else:
        youtube_transcript = None
        youtube_keywords_list = []

    return jsonify({
        "processed": True,
        "filename": text_filename,
        "coverage_score": coverage_score,
        "keywords_generated": keywords_generated_list,
        "extracted_keywords": extracted_keywords.tolist(),
        "full_text": full_text,
        "keyword_generation_time": keyword_generation_time,
        "keyword_extraction_time": keyword_extraction_time,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "youtube_transcript": youtube_transcript,
        "youtube_keywords": youtube_keywords_list,
        "youtube_keyword_generation_time": youtube_keyword_generation_time if youtube_transcript else None
    }), 200

@app.route('/processed/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True)

