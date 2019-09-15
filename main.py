import api
import solutionHelper
import os
import PIL

from fastai.vision import *
from fastai.callbacks.hooks import *

# TODO: pip install requests
# TODO: install fastai

api_key = "YOUR-API-KEY-HERE"  # TODO: Enter your API key

_infer = load_learner('/PATH/TO/FOLDER/CONTAINING/MODEL/', 'model.pkl') # TODO: Load your trained model

def main():
    result = api.init_game(api_key)
    game_id = result["gameId"]
    game_folder = '/PATH/TO/SAVE/GAME/game_' + game_id + '/'
    os.mkdir(game_folder)
    rounds_left = result['numberOfRounds']
    current_round = 1
    print("Starting a new game with id: " + game_id)
    print("The game has {} rounds and {} images per round".format(rounds_left, result["imagesPerRound"]))
    while rounds_left > 0:
        print("Starting new round, {} rounds left".format(rounds_left))
        zip_bytes = api.get_images(api_key)
        solutions = []
        # create folders for the current round of the game
        round_folder = game_folder + 'round_' + str(current_round) + '/'
        image_folder = round_folder + 'images/'
        output_folder = round_folder + 'output/'
        os.mkdir(round_folder)
        os.mkdir(image_folder)
        os.mkdir(output_folder)

        image_names = solutionHelper.save_images_to_disk(zip_bytes, image_folder)
        for name in image_names:
            solutions.append(analyze_image(name, image_folder, output_folder))

        solution_response = api.score_solution(api_key, {"Solutions": solutions})
        solutionHelper.print_errors(solution_response)
        solutionHelper.print_scores(solution_response)
        rounds_left = solution_response['roundsLeft']
        current_round += 1

def analyze_image(name, image_folder, output_folder):  # Your image recognition function here

    # Make a prediction using your trained model
    img = open_image(image_folder + name)
    mask = _infer.predict(img)[0]

    # Save mask to output_folder
    maskpath = (output_folder + name)[:-3]+'png'
    x = image2np(mask.data).astype(np.uint8)
    PIL.Image.fromarray(x).save(maskpath)

    # TODO: analyze pixel values of predicted mask and calculate percentages
    
    image_solution = {"ImageName": name,
                        "BuildingPercentage": 0,
                        "RoadPercentage": 0,
                        "WaterPercentage": 0}

    return image_solution

main()