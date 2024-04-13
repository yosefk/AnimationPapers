import cv2
import numpy as np

def visualize_custom(dict):

    source0_warp = dict['keypoints0t'][0].cpu().numpy().astype(int)
    source2_warp = dict['keypoints1t'][0].cpu().numpy().astype(int)
    source0 = dict['keypoints0'][0].cpu().numpy().astype(int)
    source2 = dict['keypoints1'][0].cpu().numpy().astype(int)
    source0_topo = dict['topo0'][0]
    source2_topo = dict['topo1'][0]
    visible01 = dict['vb0'][0].cpu().numpy().astype(int)
    visible21 = dict['vb1'][0].cpu().numpy().astype(int)

    canvas5 = np.zeros((720,720,3)) + 255


    for node, nbs in enumerate(source0_topo):
        for nb in nbs:
            if visible01[node] and visible01[nb]:
                cv2.line(canvas5, [source0_warp[node][0], source0_warp[node][1]], [source0_warp[nb][0], source0_warp[nb][1]], [0, 0, 0], 2)
    for node, nbs in enumerate(source2_topo):
        for nb in nbs:
            if visible21[node] and visible21[nb]:
                cv2.line(canvas5, [source2_warp[node][0], source2_warp[node][1]], [source2_warp[nb][0], source2_warp[nb][1]], [0, 0, 0], 2)
    im_h = cv2.hconcat([canvas5])

    
    return im_h
