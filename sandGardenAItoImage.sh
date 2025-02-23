#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_image_prompt> [--simple_mode] [--do_plots] [--external_mode]"
  exit 1
fi

input_image_prompt="$1"

shift  # Shift the arguments to process the optional flags

# Initialize optional flags
simple_mode=""
do_plots=""
external_mode=""

# Process optional flags
while (( "$#" )); do
  case "$1" in
    --simple_mode)
      simple_mode="--simple_mode"
      ;;
    --do_plots)
      do_plots="--do_plots"
      ;;
    --external_mode)
      external_mode="--external_mode"
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

#Step 1: get image
python3 sandGardenRetrieveImageAI.py "$input_image_prompt"

#Iamge to Polar
python3 imageToPolarSandGarden.py $simple_mode $do_plots $external_mode

#Send to Arduino
pio run --target upload -e nanoatmega328new
