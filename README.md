# AI Sand Garden Generator

This project converts AI-generated images into paths that can be drawn by the Hack Pack sand garden robot. The system takes a text prompt, generates an image using DALL-E, processes it into a series of coordinates, and sends these coordinates to an Arduino-controlled sand garden drawing mechanism.

## Core Components

### sandGardenAItoImage.sh
This is the main script that orchestrates the image generation and conversion process.

```bash
# Basic usage
./sandGardenAItoImage.sh "your image prompt"

# Available flags
--simple_mode     # Simplifies the traced paths
--do_plots       # Generates visualization plots
--external_mode  # Only traces the external outline
```

### imageToPolarSandGarden.py
Handles the conversion of images into polar coordinates suitable for the sand garden robot.

Key features:
- Traces image contours using OpenCV
- Converts Cartesian coordinates to polar coordinates
- Handles both external and internal contours
- Generates continuous paths for drawing
- Optional plot generation for visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AISandGarden
```

2. Install required Python packages:
```bash
pip install opencv-python numpy matplotlib fpdf shapely scikit-image
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage Example

1. Generate and process an image:
```bash
./sandGardenAItoImage.sh "a butterfly" --do_plots
```

2. The script will:
   - Generate an image using DALL-E
   - Save it as `sand_garden_image.png`
   - Convert it to polar coordinates
   - Generate visualization plots (if --do_plots is used)
   - Create a header file with coordinate data

## Optional Features

### Voice Control
The project includes `serialVoiceToSand.py` for voice-controlled operation using:
- Wake word detection
- Voice command processing
- Real-time serial communication with Arduino

There are two versions for this.  One uploaded a single script to the robot.  It pauses after the first point so you can erase the line from the center to the first point of the drawings (by shaking)

The other version uses serial communication (it needs a lot of improvements but does work).

## Notes

- The system is designed for the Chrunchlabs HackPack specific sand garden robot with known dimensions and mechanics


## License

This project is licensed under the MIT License - see the LICENSE file for details.
