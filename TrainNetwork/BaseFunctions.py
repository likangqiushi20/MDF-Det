import numpy as np
import pickle
import tensorflow as tf

def DataNormalisationZeroCentred(InputData, AverageData=None):
    """
    对输入数据进行零均值归一化。
    支持 AverageData 形状为 (C,) 或 (C, H, W) 或 (1, C, 1, 1) 等。
    """
    if AverageData is None:
        AverageData = np.mean(InputData, axis=0, keepdims=True)
        NormalisedData = InputData - AverageData
    else:
        # 将 AverageData 重塑为可广播的形状 (1, C, 1, 1)
        if AverageData.ndim == 1:
            # 形状 (C,) -> (1, C, 1, 1)
            avg = AverageData.reshape(1, -1, 1, 1)
        elif AverageData.ndim == 3:
            # 形状 (C, H, W) -> (1, C, H, W)
            avg = AverageData[np.newaxis, ...]
        elif AverageData.ndim == 4:
            avg = AverageData
        else:
            raise ValueError(f"不支持的 AverageData 形状: {AverageData.shape}")
        NormalisedData = InputData - avg

    return NormalisedData, AverageData


def ReadModels(model_folder):

    model_binary = tf.keras.models.load_model(
        model_folder + "/BinaryClassification/saved_model_2.model")
    reader_fid = open(
        model_folder + "/BinaryClassification/saved_image_norm_2.model", "rb")
    aveImg_binary = pickle.load(reader_fid)
    reader_fid.close()

    model_regression = tf.keras.models.load_model(
        model_folder + "/Regression/saved_model_3.model")
    reader_fid = open(
        model_folder + "/Regression/saved_image_norm_3.model", "rb")
    aveImg_regression = pickle.load(reader_fid)
    reader_fid.close()

    return model_binary, aveImg_binary, model_regression, aveImg_regression

