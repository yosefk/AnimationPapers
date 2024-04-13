import json, cv2, torch, sknetwork
from sknetwork.embedding import Spectral
import numpy as np

spectral = Spectral(64,  normalized=False)

class DataSample:
    def __init__(self, image_file, json_file):
        self.load(image_file, json_file)
        self.augment()

    def load(self, image_file, json_file):
        self.image = cv2.imread(image_file)
        with open(json_file) as file:
            data = json.load(file)
            self.vertex_location = np.array(data['vertex location'])
            self.connection = data['connection']

    def augment(self):
        v2d = self.vertex_location.astype(int)
        self.n = len(v2d)
        v2d = torch.from_numpy(v2d)
        imlast = self.image.shape[0] - 1 # a square image
        v2d[v2d > imlast] = imlast
        v2d[v2d < 0] = 0
        self.mask = torch.tensor([np.ones(self.n)])
        self.n = torch.tensor([self.n])
        self.keypoints = v2d.unsqueeze(0)
        topo = self.connection
        for ii in range(len(topo)):
            topo[ii].append(ii)
        adj = sknetwork.data.from_adjacency_list(topo, matrix_only=True, reindex=False).toarray()
        try:
            self.spec = torch.tensor([np.abs(spectral.fit_transform(adj))])
        except:
            print('spectral.fit_transform failed')
            self.spec = torch.tensor([np.zeros((len(adj), 64))])

        if len(self.image.shape) == 2:
            img = np.tile(self.image[...,None], (1, 1, 3))
        else:
            img = self.image[..., :3]
        self.img = (torch.from_numpy(img).permute(2, 0, 1).float() * 2 / 255.0 - 1.0).unsqueeze(0)

        self.topo = [[torch.tensor([x]) for x in ti] for ti in topo]
        
def make_model_input(sample0, sample1):
    return dict(
        keypoints0=sample0.keypoints,
        keypoints1=sample1.keypoints,
        mask0=sample0.mask,
        mask1=sample1.mask,
        ms=sample0.n,
        ns=sample1.n,
        spec0=sample0.spec,
        spec1=sample1.spec,
        image0=sample0.img,
        image1=sample1.img,
        topo0=[sample0.topo],
        topo1=[sample1.topo],
    )

#s0 = DataSample('data/ml100_norm/all/frames/chip_abe/Image0001.png', 'data/ml100_norm/all/labels/chip_abe/Line0001.json')
#s1 = DataSample('data/ml100_norm/all/frames/chip_abe/Image0005.png', 'data/ml100_norm/all/labels/chip_abe/Line0005.json')
#make_model_input(s0, s1)
