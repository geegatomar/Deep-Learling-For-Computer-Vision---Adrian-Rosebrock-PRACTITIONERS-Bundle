import h5py
import os

class HDF5_DatasetWriter:
    def __init__(self, dims, outputPath, datakey="images", bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("Path already exists ", outputPath)
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(datakey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0], ), dtype="int")
        self.bufSize = bufSize
        self.buffer = {"data":[], "labels":[]}
        self.idx = 0

    def add(self, rows, columns):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(columns)
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        self.i = self.idx + len(self.buffer["data"])
        self.data[idx : i] = self.buffer["data"]
        self.labels[idx : i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data":[], "labels":[]}




