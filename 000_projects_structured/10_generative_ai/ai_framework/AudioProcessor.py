from moviepy.editor import VideoFileClip

path_input = "./data/input"
path_output = "./data/output"

# Define the input video file and output audio file
video_file = f"{path_input}/podcast_piloto_2.mp4"
video_name = video_file.split("/")[-1].split(".")[0]
mp3_file = f"{path_output}/{video_name}.mp3"

# Load the video clip
video_clip = VideoFileClip(video_file)

# Extract the audio from the video clip
audio_clip = video_clip.audio

# Write the audio to a separate file
audio_clip.write_audiofile(mp3_file)

# Close the video and audio clips
audio_clip.close()
video_clip.close()

print("Audio extraction successful!")

from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
load_dotenv()
client = OpenAI()

def getSize(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    return size

with open(mp3_file, "rb") as audio_file:
    size_in_mb = getSize(audio_file)/(1024*1024)

    if size_in_mb >= 25:
        print("MP3 big, starting chunks of 10 minutes")
        audio = AudioSegment.from_mp3("good_morning.mp3")

        ten_minutes = 10 * 60 * 1000

        audio_len = len(audio)
        pointer_start = 0
        pointer_end = ten_minutes

        files_parts = []
        part = 1
        while pointer_start < audio_len:
            mp3_to_export = audio[pointer_start:pointer_end]
            files_parts.append(f"{mp3_file.replace('.mp3', '')}_{part}.mp3")
            mp3_to_export.export(files_parts[-1], format="mp3")

            pointer_start = pointer_end
            pointer_end = min(pointer_end + ten_minutes + 1, audio_len)
            part += 1

        for file_name in files_parts:
            transcription = ""
            with open(file_name, "rb") as audio_file_part:
                transcription_part = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file_part, 
                    response_format="text",
                    # prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."
                )
                transcription += f"{transcription_part} "
            transcription = transcription.strip()

    else:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="text",
            # prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."
        )

with open(f"{path_output}/{video_name}.txt", "w") as f_out:
    if isinstance(transcription, str):
        f_out.write(transcription)
    else:
        f_out.write(transcription.text)


import os
from pydub import AudioSegment
from pydub.playback import play


# cwd = os.getcwd()
sound = AudioSegment.from_file(f"{path_output}/tts_podcast_piloto_2.mp3", format="mp3")

# print(sound.frame_rate)
# shift the pitch down by half an octave (speed will decrease proportionally)
octaves = +0.05

new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))

new_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})

new_sound.export(f"{path_output}/tts_podcast_piloto_2_new.mp3", format="mp3")