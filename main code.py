import streamlit as st
import os
from google.cloud import speech
from google.cloud import texttospeech


from moviepy.editor import VideoFileClip, AudioFileClip

# Set up Azure OpenAI credentials
AZURE_OPENAI_API_KEY = os.environ.get('22ec84421ec24230a3638d1b51e3a7dc')
AZURE_OPENAI_ENDPOINT_URL = os.environ.get('https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview')

# Set up Google Cloud credentials
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

def transcribe_audio(file_path):
    # Transcribe audio using Google Cloud Speech-to-Text
    client = speech.SpeechClient()
    with open(file_path, 'rb') as audio_file:
        audio = speech.RecognitionAudio(content=audio_file.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code='en-US'
    )
    response = client.recognize(config, audio)
    transcription = ''
    for result in response.results:
        transcription += result.alternatives[0].transcript
    return transcription

def correct_transcription(transcription):
    # Correct transcription using Azure OpenAI GPT-4o
    credential = AzureKeyCredential(AZURE_OPENAI_API_KEY)
    client = OpenAIServiceClient(credential, AZURE_OPENAI_ENDPOINT_URL)
    response = client.openai_api.create_completion(
        deployment_name='Hey',
        display_name='hi',
        input_text=transcription,
        max_tokens=2048,
        temperature=0.5
    )
    corrected_transcription = response.result[0].text
    return corrected_transcription

def generate_audio(transcription):
    # Generate audio using Google Cloud Text-to-Speech
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=transcription)
    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        name='en-US-Wavenet-J'
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    return response.audio_content

def replace_audio(video_file_path, audio_file_path):
    # Replace audio in video file using moviepy
    video_clip = VideoFileClip(video_file_path)
    audio_clip = AudioFileClip(audio_file_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile('output.mp4')

def main():
    st.title('Video Audio Replacement')
    st.write('Upload a video file to replace its audio with an AI-generated voice.')
    video_file = st.file_uploader('Upload a video file', type=['mp4'])
    if video_file:
        video_file_path = video_file.name
        with open(video_file_path, 'wb') as f:
            f.write(video_file.read())
        transcription = transcribe_audio(video_file_path)
        corrected_transcription = correct_transcription(transcription)
        audio_content = generate_audio(corrected_transcription)
        audio_file_path = 'output.mp3'
        with open(audio_file_path, 'wb') as f:
            f.write(audio_content)
        replace_audio(video_file_path, audio_file_path)
        st.write('Audio replaced successfully!')
        st.video('output.mp4')

if __name__ == '__main__':
    main()
