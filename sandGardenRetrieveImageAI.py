
import openai
import os
import requests
import argparse
from os import environ
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
parser = argparse.ArgumentParser(description='Generate an image using OpenAI.')

# Add a command-line argument for the prompt
parser.add_argument('prompt', type=str, help='Prompt for generating the image')

def getSandGardenAIGeneratedImageURL(image_prompt):
    try:

        client = OpenAI()

        #openAIImagePrompt = "Generate a black and white vector art like image of a simple children's illustration featuring " + image_prompt + ", suitable for svg conversion, focusing on smooth, clean and crisp lines with a bold, consistent stroke width, featuring a central centered object on a white background but including all the elements of the prompt."

        # This one is ok
        #openAIImagePrompt = "Generate a minamalist black and white single line cartoon drawing featuring a " + image_prompt + ", focusing on smooth, clean and crisp lines with a bold, consistent stroke width, featuring a central centered object on a white background but including all the elements of the prompt."

        #openAIImagePrompt = "Generate a minamalist black and white silhouette cartoon drawing featuring a " + image_prompt + ", featuring a central centered object."

        # This prompt is pretty good
        #openAIImagePrompt = "Generate an extremely basic black and white cartoon line drawing featuring a " + image_prompt + ", featuring a central centered object on a white background"
        #openAIImagePrompt = "Generate a black and white silhouette cartoon illustration featuring a " + image_prompt + ", featuring a single central centered object with no inner portions and no background."
        #openAIImagePrompt = "minimal B&W cartoon illustration, " + image_prompt + ", svg, flat minimal line vector design, white background" #think about removing "illustration" may could think about adding "profile"
        openAIImagePrompt = "minimal B&W, exterior outline of a simple cartoon " + image_prompt + ", svg, flat minimal line vector design, white background, featuring a single centered object" 


        print("Prompt Sent to Dall-E ", openAIImagePrompt)

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


try:
    args = parser.parse_args()
    print("The prompt was: " + args.prompt)

    openai_image_url = getSandGardenAIGeneratedImageURL(args.prompt)

    # Send an HTTP GET request to the URL
    imageData = requests.get(openai_image_url)

    # Check if the request was successful (status code 200)
    if imageData.status_code == 200:
        with open('sand_garden_image.png', 'wb') as image_file:
            image_file.write(imageData.content)
            print("Image saved to sand_garden_image.png")
    else:
        print("Failed to fetch the image.")

except Exception as e:
    print("Unknown error: ", e)
