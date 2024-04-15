import cv2
import numpy as np
import json

class Sample:
    def __init__(self):
        self.vertexes = []
        self.conn = {} 
        self.image = np.ones((720,720,3))*255
        self.set_cursor((0,0))

    def set_cursor(self, pos):
        self.cursor = pos
        self.drawing = False

    def connect(self, vertex_ind0, vertex_ind1):
        self.conn.setdefault(vertex_ind0, list()).append(vertex_ind1)
        self.conn.setdefault(vertex_ind1, list()).append(vertex_ind0)


    def draw_line_to(self, pos, steps=100):
        x0,y0 = self.cursor
        x1,y1 = pos
        for i in range(1,steps+1):
            t = i/steps
            x = x0*(1-t) + x1*t
            y = y0*(1-t) + y1*t
            if self.drawing:
                self.connect(len(self.vertexes), len(self.vertexes)-1)
            assert([int(x),int(y)] not in self.vertexes)
            self.vertexes.append([int(x),int(y)])
            self.drawing = True
        self.cursor = pos

    def draw_image(self):
        def toint(s): return [int(x) for x in s]
        for index,connected in self.conn.items():
            for index2 in connected:
                cv2.line(self.image, toint(self.vertexes[index]), toint(self.vertexes[index2]), [0, 0, 0], 1)

    def save(self, prefix):
        cv2.imwrite(prefix+'.png', self.image)
        d = {'vertex location':self.vertexes, 'connection':list([self.conn.get(index,[]) for index in range(len(self.vertexes))])}
        with open(prefix+'.json','w') as f:
            json.dump(d, f, indent=2)

