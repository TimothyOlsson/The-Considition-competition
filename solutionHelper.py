import zipfile
import io
import os


def save_images_to_disk(zip_bytes, path):
    """
    Opens a zip file and stores the images to the specified path. Returns a list of names of the images
    :param zip_bytes: A byte array representing a zip file
    :param path: The path to the folder to unpack the images to
    :return: An array of strings
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))
    zip_file.extractall(path)

    return zip_file.namelist()


def clean_images_from_folder(path):
    """
    Clears the specified folder of all .jpg images
    :param path: The path to the folder to clear
    """
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path) and ".jpg" in file_name:
                os.unlink(file_path)
        except Exception as e:
            print(e)


def print_errors(response):
    """
    Prints any errors encountered in the solution
    :param response: The solution response to check
    """
    if response["errors"] is None or len(response["errors"]) == 0:
        return
    print("Encountered some errors with the solution:")
    for error in (response["errors"]):
        print(error)


def print_scores(response):
    """
    Prints the scores for this round
    :param response: The solution response to check
    """
    print("Total score: {}".format(response["totalScore"]))
    for score in response["imageScores"]:
        print("Image {:>25s} got a score of {}".format(score["imageName"], score["score"]))