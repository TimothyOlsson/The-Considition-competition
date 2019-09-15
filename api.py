import sys
import time
import requests
from requests import RequestException

base_api_path = "https://api.theconsidition.se/api/game/"


def init_game(api_key):
    """
    Closes any active game and starts a new one, returning an object with information about the game
    :param api_key: The teams api key
    :return: {gameId, numberOfRounds, imagesPerRound}
    """
    try:
        response = requests.get(base_api_path + "init", headers={'x-api-key': api_key})
        if response.status_code == 200:
            return response.json()

        print("Fatal Error: could not start game")
        print(str(response.status_code) + " " + response.reason + ": " + response.text)
        sys.exit(1)
    except RequestException as e:
        print("Fatal Error: could not start game")
        print("Something went wrong with the request: " + str(e))
        sys.exit(1)


def get_images(api_key):
    """
    Gets a zip-file containing images for the current round and returns a bytearray representing the file.
    Tries fetching the zip file three times
    :param api_key: The teams api key
    :return: A byte array representing the zip file
    """
    tries = 1
    while tries <= 3:
        try:
            print("Fetching images...")
            response = requests.get(base_api_path + "images", headers={'x-api-key': api_key}, stream=True)
            if response.status_code == 200:
                return response.content
            print(str(response.status_code) + " " + response.reason + ": " + response.text)
            print("Attempt {} failed, waiting 2 sec and attempting again.".format(tries))
            time.sleep(2)
            tries += 1
        except RequestException as e:
            print("Something went wrong with the request: " + str(e))
            print("Attempt {} failed, waiting 2 sec and attempting again.".format(tries))
            time.sleep(2)
            tries += 1
    print("Fatal Error: could not fetch images")
    sys.exit(2)


def score_solution(api_key, solution):
    """
    Posts the solution for evaluation. Returns a summary of the score and game state.
    Tries to submit the solution three times
    :param api_key: The teams api key
    :param solution: The solution for this round of images
    :return: {totalScore, imageScores[], roundsLeft, errors[]}
    """
    tries = 1
    while tries <= 3:
        try:
            print("Submitting solution...")
            response = requests.post(base_api_path + "solution", headers={'x-api-key': api_key}, json=solution)
            if response.status_code == 200:
                return response.json()
            print(str(response.status_code) + " " + response.reason + ": " + response.text)
            print("Attempt {} failed, waiting 2 sec and attempting again.".format(tries))
            time.sleep(2)
            tries += 1
        except RequestException as e:
            print("Something went wrong with the request:" + str(e))
            print("Attempt {} failed, waiting 2 sec and attempting again.".format(tries))
            time.sleep(2)
            tries += 1
    print("Fatal Error: could not submit solution")
    sys.exit(3)