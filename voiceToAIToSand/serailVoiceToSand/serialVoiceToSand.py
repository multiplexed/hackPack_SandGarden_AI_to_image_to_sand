# Description: This script listens for a wake word, then listens for a voice command, sends the voice command to OpenAI to generate an image, then converts the image to polar coordinates and sends the polar coordinates to the Sand Garden drawing robot.
import openai
import os
import pvcobra
import pvleopard
import pvporcupine
import pyaudio
import random
import struct
import sys
import textwrap
import threading
import time
import requests
import subprocess
import imageToPolarSandGarden
import serial

from os import environ

from colorama import Fore, Style
from pvleopard import *
from pvrecorder import PvRecorder
from threading import Thread, Event
from time import sleep
from openai import OpenAI
import argparse




audio_stream = None
cobra = None
pa = None
porcupine = None
recorder = None
wav_file = None


openai.api_key = os.getenv("OPENAI_API_KEY")
pv_access_key= os.getenv("PICOVOICE_API_KEY")

next_event = threading.Event()



def read_serial(ser):
    global next_event
    while True:
        try:
            if ser.is_open and ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()
                print(f"Arduino: {line}")
                if line == "next":
                    next_event.set()
            time.sleep(0.1)  # Add small delay to prevent busy-waiting
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            break
        except OSError as e:
            print(f"Serial read error: {e}")
            # Don't break here, try to continue reading
            time.sleep(1)  # Wait a bit before trying again

def wake_word():

    porcupine = pvporcupine.create(keyword_paths=['Sand-Garden_en_raspberry-pi_v3_0_0.ppn'],
                            access_key=pv_access_key,
                            sensitivities=[0.6], #from 0 to 1.0 - a higher number reduces the miss rate at the cost of increased false alarms
                                   )

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)

    wake_pa = pyaudio.PyAudio() #should I try to replace with pvrecord?

    porcupine_audio_stream = wake_pa.open(
                    rate=porcupine.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=porcupine.frame_length)

    Detect = True

    while Detect:
        porcupine_pcm = porcupine_audio_stream.read(porcupine.frame_length)
        porcupine_pcm = struct.unpack_from("h" * porcupine.frame_length, porcupine_pcm)

        porcupine_keyword_index = porcupine.process(porcupine_pcm)

        if porcupine_keyword_index >= 0:
            print(Fore.GREEN + "\nWake word detected\n")
            porcupine_audio_stream.stop_stream
            porcupine_audio_stream.close()
            porcupine.delete()
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            Detect = False

def listen():

    cobra = pvcobra.create(access_key=pv_access_key)

    listen_pa = pyaudio.PyAudio()

    listen_audio_stream = listen_pa.open(
                rate=cobra.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=cobra.frame_length)

    print("Listening...")

    while True:
        listen_pcm = listen_audio_stream.read(cobra.frame_length)
        listen_pcm = struct.unpack_from("h" * cobra.frame_length, listen_pcm)

        if cobra.process(listen_pcm) > 0.3:
            print("Voice detected")
            listen_audio_stream.stop_stream
            listen_audio_stream.close()
            cobra.delete()
            break

def detect_silence():

    cobra = pvcobra.create(access_key=pv_access_key)

    silence_pa = pyaudio.PyAudio()

    cobra_audio_stream = silence_pa.open(
                    rate=cobra.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=cobra.frame_length)

    last_voice_time = time.time()

    while True:
        cobra_pcm = cobra_audio_stream.read(cobra.frame_length)
        cobra_pcm = struct.unpack_from("h" * cobra.frame_length, cobra_pcm)

        if cobra.process(cobra_pcm) > 0.2:
            last_voice_time = time.time()
        else:
            silence_duration = time.time() - last_voice_time
            if silence_duration > 1.3:
                print("End of query detected\n")
                cobra_audio_stream.stop_stream
                cobra_audio_stream.close()
                cobra.delete()
                last_voice_time=None
                break

class Recorder(Thread):
    def __init__(self):
        super().__init__()
        self._pcm = list()
        self._is_recording = False
        self._stop = False

    def is_recording(self):
        return self._is_recording

    def run(self):
        self._is_recording = True

        recorder = PvRecorder(device_index=-1, frame_length=512)
        recorder.start()

        while not self._stop:
            self._pcm.extend(recorder.read())
        recorder.stop()

        self._is_recording = False

    def stop(self):
        self._stop = True
        while self._is_recording:
            pass

        return self._pcm


def getAIGeneratedImageURL(image_prompt):
    try:

        #get openAI image from dall-e
        client = OpenAI()

        gptQueryTransform = "Given the below prompt text, provide a prompt suitable for a request to the dall-E image generator to generate an image optimized for conversion to a black and white svg file that will be converted into gcode for a drawing robot.  The image should appeal to a chile and feature smooth, crisp, and clean shapes and lines.  Prompt: " + image_prompt

        gptQueryCleanUp = "Clean up the below prompt relating to asking for a particular image to be generated so it strips the text down to just the exact image requested without text like 'draw me a picture of', 'a picture,' or similar phrases.  Prompt: " + image_prompt

        completion = client.chat.completions.create(
            #model="gpt-3.5-turbo",
            model="chatgpt-4o-latest",
            messages=[
                #{"role": "system", "content": "You are a technical assistant skilled in generating image prompts for an AI image generator."},
                {"role": "user", "content": gptQueryCleanUp}
            ]
        )


        image_prompt_gpt = completion.choices[0].message.content#['message']['content']
        print("Image Prompt: ", image_prompt_gpt)

        #image_prompt1 = "Generate a simple black and white cartoon vector art illustration featuring " + image_prompt_gpt + ", suitable for svg conversion, focusing on smooth, clean and crisp lines with a bold, consistent stroke width, featuring a central centered object on a white background,"
        openAIImagePrompt = "minimal B&W, exterior outline of a simple cartoon " + image_prompt_gpt + ", svg, flat minimal line vector design, white background, featuring a single centered object,fully surrounded by white" 

       
        response = client.images.generate(
          prompt= openAIImagePrompt,
          n=1,
          size="512x512"
        )

        openai_image_url = response.data[0].url
        print("URL: ", openai_image_url)
        return openai_image_url


    except openai.BadRequestError as e:
        print("\nBad Request error.  Please try a new prompt.")


    except openai.APIError as e:
        print("\nThere was an API error.  Please try again in a few minutes.")


    except openai.Timeout as e:
        print("\nYour request timed out.  Please try again in a few minutes.")


    except openai.RateLimitError as e:
        print("\nYou have hit your assigned rate limit.")


    except openai.APIConnectionError as e:
        print("\nI am having trouble connecting to the API.  Please check your network connection and then try again.")


    except openai.AuthenticationError as e:
        print("\nYour OpenAI API key or token is invalid, expired, or revoked.  Please fix this issue and then restart my program.")

    except openai.ServiceUnavailableError as e:
        print("\nThere is an issue with OpenAI’s servers.  Please try again later.")



def main():
    parser = argparse.ArgumentParser(description='Process an image and convert to polar coordinates.')
    parser.add_argument('--external_mode', action='store_true', help='Use external mode for tracing only the external outline of the image')
    args = parser.parse_args()
    ser = serial.Serial('/dev/ttyUSB1', 9600)  # Adjust the port name as needed
    time.sleep(2)  # Wait for the serial connection to initialize
    # Start the serial reading thread
    serial_thread = threading.Thread(target=read_serial, args=(ser,))
    serial_thread.daemon = True
    serial_thread.start()
    

    print("External Mode: ", args.external_mode)
    try:

        o = create(
            access_key=pv_access_key,
            enable_automatic_punctuation = True,
            )

        event = threading.Event()

        
        while True:

            try:
                wake_word()
                #ListeningState
                recorder = Recorder()
                recorder.start()
                listen()
                detect_silence()
                transcript, words = o.process(recorder.stop())
                recorder.stop()
            
                
                print(transcript)
                recorder.stop()
                o.delete
                recorder = None

            
                openai_image_url = getAIGeneratedImageURL(transcript)

                # Send an HTTP GET request to the URL
                imageData = requests.get(openai_image_url)

                # Check if the request was successful (status code 200)
                if imageData.status_code == 200:
                    with open('sand_garden_image.png', 'wb') as image_file:
                        image_file.write(imageData.content)
                    print("Image saved to local_image.png")
                else:
                    print("Failed to fetch the image.")

               
                saved_image_file = 'sand_garden_image.png'  # Make sure this file is present or change accordingly
        
                #Convert image to polar coordinates
                externalcontour, external_contour_simplified, inner_contours_simplified = imageToPolarSandGarden.trace_image(saved_image_file, simple_mode=False)

            
                if args.external_mode:
                    print("External Mode: Only returns points for outer contour")
                    continuous_path = [tuple(point[0]) for point in external_contour_simplified]
                else:
                    continuous_path = imageToPolarSandGarden.create_continuous_trace_path(external_contour_simplified, inner_contours_simplified)
        
                print("Length of continous path: ", len(continuous_path))
                print("Type of continous path: ", type(continuous_path))
        
                polar_coords = imageToPolarSandGarden.convert_to_polar(continuous_path)


                #batch_size = 100
                batch_size = 1
                for i in range(0, len(polar_coords), batch_size):
                    batch = polar_coords[i:i + batch_size]
                    current_batch_size = len(batch)
                    try:
                        # Send the batch size first
                        ser.write(f"SIZE:{current_batch_size}\n".encode())
                        print(f"Sending batch of {current_batch_size} points")
                        time.sleep(0.2)  # Give Arduino time to process the size
                        
                        # Send the points
                        for point in batch:
                            ser.write(f"{point[0]},{point[1]}\n".encode())
                            print(f"Sent: {point[0]},{point[1]}")
                            time.sleep(0.2)  # Small delay to ensure data is sent properly
                        next_event.wait()  # Wait for the "next" message
                        next_event.clear()  # Clear the event for the next iteration
                    except serial.SerialException as e:
                        print(f"Serial error during transmission: {e}")
                        break
                

                print("All points sent")

                

            except Exception as e:
                print("Unknown error: ", e)
                ser.close()


    except KeyboardInterrupt:
        print("\nExiting Sand Garden Voice Assistant")
        o.delete
        ser.close()


if __name__ == '__main__':
    main()