import pytest

from . import (
    read_yaml,
    get_data,
    load_model
)


def test_read_yaml_no_file():
    with pytest.raises(FileNotFoundError):
        read_yaml('docs')
        read_yaml('docs/f.yaml')


def test_read_yaml():
    assert read_yaml('src/qcsfs/neural_network/nn_config.yaml')\
        .get('TRAINING', {}).get('LABELS', None) == 'Good,Damaged'


def test_get_data_no_imgs():
    with pytest.raises(FileNotFoundError):
        get_data('tests', ['Good', 'Damaged'], 'Good', (224, 224))


def test_load_model_no_model():
    with pytest.raises(OSError):
        load_model('tests')
