import os
import random
import PySimpleGUI as sg

from neural_network import nn_training


def create_img_list() -> list[tuple[str | os.PathLike, str]]:
    """
    This function creates a list of paths to the images of the screws and their labels.
    #### Returns:
        (list[tuple[str | os.PathLike, str]]): A shuffled list of paths to the images of the screws and their labels.
    """
    screws = []
    paths = ('data/screws/test/good', 'data/screws/test/damaged')
    for path_ in os.listdir(paths[0]):
        screws.append((os.path.join(paths[0], path_), 'Good'))
    for path_ in os.listdir(paths[1]):
        screws.append((os.path.join(paths[1], path_), 'Damaged'))
    random.shuffle(screws)

    return screws


def is_data_present() -> bool:
    """
    This function checks if screw images are present.
    #### Returns:
        (bool)
    """
    dirs = ['data/screws/test/good', 'data/screws/test/damaged']
    extensions = ['.png', '.jpg', '.jpeg']
    for dir in dirs:
        for file in os.listdir(dir):
            if os.path.splitext(file)[-1].lower() in extensions:
                return True
    return False


def is_model_present() -> bool:
    """
    This function checks if neural network model is present.
    #### Returns:
        (bool)
    """
    dirs = ['data/models']
    extensions = ['.pb']
    for dir in dirs:
        for file in os.listdir(dir):
            if os.path.splitext(file)[-1].lower() in extensions:
                return True
    return False


def is_result_present() -> bool:
    """
    This function checks if measurement results are present.
    #### Returns:
        (bool)
    """
    dirs = ['data/measuring_results']
    for dir in dirs:
        if 'measurements_axis.csv' in os.listdir(dir) and 'measurements_rect.csv' in os.listdir(dir):
            return True
    return False


def guard_popup() -> tuple[bool, bool]:
    """
    This function creates guard popup,
    which prevents the user from accessing the main application without the neural network model and screw images.
    #### Returns:
        (tuple[bool]): Boolean values indicating the presence of the model and images.
    """
    is_data = is_data_present()
    is_model = is_model_present()
    train = False
    if is_data and is_model:
        msg1 = 'Images found.'
        msg2 = 'Model found. You can press the exit button.\nAfter a moment, the main application window\nshould appear.'
    elif is_data and not is_model:
        msg1 = 'Images found.'
        msg2 = 'No model found. Add model in the directory\ndata/models and'\
            ' restart the app (or create model).\nIn case of further problems, see README.md.'
        train = True
    elif not is_data and is_model:
        msg1 = 'No images found. Add images in the directories\ndata/screws/good and/or data/screws/damaged and\n'\
            'restart the app. In case of further problems,\nsee README.md.'
        msg2 = 'Model found.'
    else:
        msg1 = 'No images found. Add images in the directories\ndata/screws/good and/or data/screws/damaged and\n'\
            'restart the app. In case of further problems,\nsee README.md.'
        msg2 = 'No model found. Add model in the directory\ndata/models and'\
            ' restart the app. In case of further\nproblems, see README.md.'

    font_size = 12
    col = [
        [sg.Text('Have images of the screws been found?', font=('Arial', font_size))],
        [sg.Text(key='-OUTPUT1-', justification='left',
                 font=('Arial', font_size), text=msg1)],
        [sg.Text('Has a neural network model been found?', font=('Arial', font_size))],
        [sg.Text(key='-OUTPUT2-', justification='left',
                 font=('Arial', font_size), text=msg2)],
        [sg.Text(key='-OUTPUT3-', text='To create neural network model click button bellow:',
                 font=('Arial', font_size), visible=train)],
        [sg.Button(button_text='Create model', key='-BUTTON1-',
                   font=('Arial', font_size), visible=train)],
        [sg.Exit('Exit', font=('Arial', font_size))]
    ]
    pop_layout = [
        [sg.VPush()],
        [sg.Push(), sg.Column(col,  element_justification='c'), sg.Push()],
        [sg.VPush()]
    ]

    start_popup = sg.Window("Checking...", pop_layout, margins=(10, 10),
                            size=(450, 350), finalize=True)

    while True:
        event, _ = start_popup.read()  # type: ignore

        if event in ('Exit', sg.WIN_CLOSED):
            break

        if event in '-BUTTON1-':
            msg = 'Creating neural network model in progress. See terminal output for more information.'
            start_popup['-BUTTON1-'].update(visible=False)
            start_popup['-OUTPUT2-'].update(value=msg)
            start_popup['-OUTPUT3-'].update(visible=False)
            nn_training.create_model()
            is_model = is_model_present()
            if is_model:
                msg = 'Model found. You can press the exit button.\nAfter a moment, the main application window\nshould appear.'
                start_popup['-OUTPUT2-'].update(value=msg)

    start_popup.close()

    return is_data, is_model
