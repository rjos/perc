from .keel_dataset import load_from_file
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

__path_to_raw_data = 'data/raw'


def build(name):
    """
    Load a KEEL dataset 
    """

    path = os.path.join(__path_to_raw_data, name + '.dat')
    data = load_from_file(path)

    x, y = data.get_data_target()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return x, y


def normalize(x, scale=(0, 1)):
    """
    Normalize dataset attributes
    """

    mms = MinMaxScaler(feature_range=scale)
    return mms.fit_transform(x)
