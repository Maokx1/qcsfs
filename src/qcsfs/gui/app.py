# Based on: https://www.pysimplegui.org/en/latest/cookbook/

import io
from itertools import cycle
import os
import sys
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import PySimpleGUI as sg

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gui import gui_utils
from measure_screws import measure
from neural_network import nn_training, nn_predicting
from spc import spc_charts


def main():
    # The default measurement method is the axis method
    METHOD_STATUS = 'Axis method'
    CAN_CHANGE_METHOD = False

    # Main layout
    sg.theme('DarkBlue11')
    col_1 = [
        [sg.Button('Change the measurement method')],
        [sg.Text('Original image:')],
        [sg.Image(key='-IMAGE1-', size=(512, 512))],
        [sg.Text('Ground truth: '), sg.Text(key='-OUTPUT1-')],
        [sg.Button('Display image/Next image')],
        [sg.Button('Show all measurements')]
    ]
    col_2 = [
        [sg.Text('Measurement method: ', pad=(4, 6)),
         sg.Text(key='-OUTPUT2-', text=METHOD_STATUS)],
        [sg.Text('Processed image:')],
        [sg.Image(key='-IMAGE2-', size=(512, 512))],
        [sg.Text('Predicted label: '), sg.Text(key='-OUTPUT3-')],
        [sg.Text('Length of the screw: ', pad=(4, 6)),
         sg.Text(key='-OUTPUT4-', pad=(4, 6))],
        [sg.Exit('Exit')]
    ]
    main_layout = [
        [sg.Column(col_1), sg.Column(col_2),]
    ]

    is_data, is_model = gui_utils.guard_popup()

    if is_data and is_model:
        cycle_img_list = cycle(gui_utils.create_img_list())
        _curr = next(cycle_img_list)
        model = nn_predicting.load_model('data/models')

        if os.path.isfile('nn_config.yaml'):
            config_values = nn_training.read_yaml('nn_config.yaml')
        else:
            top_dir = os.path.dirname(os.path.dirname(__file__))
            config_values = nn_training.read_yaml(
                os.path.join(top_dir, 'neural_network/nn_config.yaml'))

        try:
            labels = config_values['TRAINING']['LABELS'].replace(' ', '')\
                .split(',')
            img_shape = config_values['TRAINING']['IMG_SHAPE'].split(',')
            img_shape = tuple([int(i) for i in img_shape])
            membership_boundary = float(
                config_values['TRAINING']['MEMBERSHIP_BOUNDARY'])
        except KeyError:
            traceback.print_exc()
            exit('Some default keys were changed in yaml file without updating code.')
        except ValueError:
            traceback.print_exc()
            exit('The newly entered data in the yaml file does not meet the requirements stated in the comments.')

        # Main event loop
        window = sg.Window("QualityControlSystemForScrews",
                           main_layout, margins=(20, 20), size=(1100, 715))
        while True:
            event, _ = window.read()  # type: ignore

            if event in ("Exit", sg.WIN_CLOSED):
                break

            if event == 'Change the measurement method':
                if METHOD_STATUS == 'Axis method':
                    METHOD_STATUS = 'Rectangle method'
                    window["-OUTPUT2-"].update(value=METHOD_STATUS)
                    if CAN_CHANGE_METHOD:
                        window.write_event_value('Measure', 0)
                else:
                    METHOD_STATUS = 'Axis method'
                    window["-OUTPUT2-"].update(value=METHOD_STATUS)
                    if CAN_CHANGE_METHOD:
                        window.write_event_value('Measure', 0)

            if event == 'Display image/Next image':
                # loading image to buffer
                _curr = next(cycle_img_list)  # type: ignore
                display_image = Image.open(_curr[0])  # type: ignore
                display_image.thumbnail((512, 512))
                bio = io.BytesIO()
                display_image.save(fp=bio, format="PNG")
                window["-IMAGE1-"].update(data=bio.getvalue())
                data = nn_training.get_data(_curr[0], labels,
                                            _curr[1], img_shape)  # type: ignore
                label = 'Good' if data[0][1] == 0 else 'Damaged'
                gt, predicted = nn_predicting.predict(model, np.asarray([data[0][0]]), label,  # type: ignore
                                                      labels, membership_boundary)  # type: ignore
                print(gt, predicted)
                window["-OUTPUT1-"].update(value=gt)
                window["-OUTPUT3-"].update(value=predicted)
                if predicted == 'Good':
                    CAN_CHANGE_METHOD = True
                    window.write_event_value('Measure', 0)
                else:
                    CAN_CHANGE_METHOD = False
                    window["-IMAGE2-"].update(data='', size=(512, 512))
                    window["-OUTPUT4-"].update(value='')

            if event == 'Measure':
                img = cv2.imread(_curr[0])  # type: ignore
                img = cv2.resize(img, (512, 512))
                if METHOD_STATUS == 'Axis method':
                    axis_img, length = measure.axis_method(img)  # type: ignore
                    processed_image = Image.fromarray(axis_img)
                    bio = io.BytesIO()
                    processed_image.save(bio, format="PNG")
                    window["-IMAGE2-"].update(data=bio.getvalue(),
                                              size=(512, 512))
                    window["-OUTPUT4-"].update(value=round(length, 3))
                else:
                    rect_img, length = measure.rect_method(img)  # type: ignore
                    processed_image = Image.fromarray(rect_img)
                    bio = io.BytesIO()
                    processed_image.save(bio, format="PNG")
                    window["-IMAGE2-"].update(data=bio.getvalue(),
                                              size=(512, 512))
                    window["-OUTPUT4-"].update(value=round(length, 3))

            if event == 'Show all measurements':
                pd.set_option('display.max_rows', None)
                if not gui_utils.is_result_present():
                    # Breaking this into 2 threads was ~37% slower.
                    # This may be due to an implementation error or due to data racing (loading images from the same directory).
                    measure.log_measurements_rect('data/screws/test/good')
                    measure.log_measurements_axis('data/screws/test/good')

                if METHOD_STATUS == 'Axis method':
                    try:
                        df = pd.read_csv(
                            'data/measuring_results/measurements_axis.csv')

                        spc_charts.create_xbar_s_chart(list(df['length']),
                                                       num_of_subgroups=10)
                    except ValueError or pd.errors.EmptyDataError:
                        traceback.print_exc()
                        exit('Check that the csv files in '
                             'data/measuring_results are not empty.')
                else:
                    try:
                        df = pd.read_csv(
                            'data/measuring_results/measurements_rect.csv')
                        spc_charts.create_xbar_s_chart(list(df['length']),
                                                       num_of_subgroups=10)
                    except ValueError or pd.errors.EmptyDataError:
                        traceback.print_exc()
                        exit('Check that the csv files in '
                             'data/measuring_results are not empty.')
                # Table popup
                table_layout = [[sg.Table(df.values.tolist(),
                                          headings=df.columns.tolist(), justification='center', key='-TABLE-')]]
                table = sg.Window(layout=table_layout, title=f'All good screws({METHOD_STATUS})',
                                  size=(350, 260), font=('Arial', 12), finalize=True)
                while True:
                    e, v = table.read()  # type: ignore
                    if e == sg.WIN_CLOSED:
                        break
                plt.show()

        window.close()


if __name__ == '__main__':
    main()
