import json
import base64
import numpy as np
import matplotlib.pyplot as plt


class DataDecoder:
    def __init__(self, filename):
        self.data = ''
        with open(filename, 'r') as file:
            self.data = file.read()
        self.structed_data = json.loads(self.data)

    def getRawData(self, averaging_num=0, data_num=0, channel_num=0):
        key = f"averaging_num_{averaging_num}"
        if key in self.structed_data:
            measurements = self.structed_data[key]
            for measurement in measurements:
                if measurement.get('data_num') == data_num:
                    for channel_data in measurement.get('channel_data', []):
                        if channel_data.get('channel_num') == channel_num:
                            return channel_data.get('channel_data')
        return None

    def getDataDecoded(self, averaging_num=0, data_num=0, channel_num=0, points=10):
        rawData = self.getRawData(averaging_num, data_num, channel_num)
        edata = base64.b64decode(rawData.encode('utf-8'))
        arr = np.frombuffer(edata, dtype=np.int16, count=points, offset=0)
        return arr

    def getDataScaled(self, averaging_num=0, data_num=0, channel_num=0, points=10, v_range=0.5):
        decodedData = self.getDataDecoded(averaging_num, data_num, channel_num, points)
        return v_range * decodedData / 32768

    def getDataRate(self, averaging_num=1, data_num=0):
        key = f"averaging_num_{averaging_num}"
        if key in self.structed_data:
            measurements = self.structed_data[key]
            for measurement in measurements:
                if measurement.get('data_num') == data_num:
                    return measurement.get('measurement_rate')

        return None


    def getDataPoints(self, averaging_num=1, data_num=0):
        key = f"averaging_num_{averaging_num}"
        if key in self.structed_data:
            measurements = self.structed_data[key]
            for measurement in measurements:
                if measurement.get('data_num') == data_num:
                    return measurement.get('measurement_points')

        return None
