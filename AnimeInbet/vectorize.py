import imageio, sys
import cv2
import numpy as np
import sample
sys. setrecursionlimit(1000000) 
import skimage.morphology
import os

def bounding_box_of_black_pixels(array):
    # Find indices of black pixels
    black_pixels_indices = np.argwhere(array == 0)

    if len(black_pixels_indices) == 0:
        return None  # No black pixels found

    # Get coordinates of the bounding box
    min_row, min_col = np.min(black_pixels_indices, axis=0)
    max_row, max_col = np.max(black_pixels_indices, axis=0)

    # Return bounding box coordinates
    return (min_row, min_col), (max_row, max_col)

newflood = False
last_coord = (0,0)
def vectorize(infile,outprefix):
  global last_coord
  last_coord = 0,0
  arr = to720x720bw(imageio.imread(infile))
  imageio.imwrite('arr.png', arr)
  w,h = arr.shape
  print(infile,outprefix,arr.shape)
  arr = skimage.morphology.skeletonize(arr==0)
  imageio.imwrite('sk.png', arr.astype(np.uint8)*255)
  arr = 255 - arr.astype(np.uint8)*255
  
  s = sample.Sample()
  
  def black(val): return val==0
  
  def inbounds(x,y):
      return x>=0 and y>=0 and x<w and y<h
  
  def sqdist(x,y,last_coord):
      x1,y1=last_coord
      return (x-x1)**2+(y-y1)**2
  
  def flood_fill_indexes(x,y,last_coord):
      global newflood
      arr[x,y] = 255
      sd = sqdist(x,y,last_coord)
      if sd>1:
          toconn=None
          if s.cursor != (last_coord[1],last_coord[0]):
              s.set_cursor((last_coord[1],last_coord[0])) # TODO: fix topology here 
              toconn=last_coord

          last_coord = x,y
          s.draw_line_to((y,x),steps=1)
          if not newflood and toconn:
              try:
                  print('CONNECTED',s.vertexes.index([toconn[1],toconn[0]]), len(s.vertexes)-1)
                  s.connect(s.vertexes.index([toconn[1],toconn[0]]), len(s.vertexes)-1)
              except:
                  print('NOT!')
      newflood = False
      for nx in range(x-1,x+2):
          for ny in range(y-1,y+2):
              if inbounds(nx,ny) and black(arr[nx,ny]):
                  flood_fill_indexes(nx,ny,last_coord)
  
  while np.any(black(arr)):
      xs, ys = np.where(black(arr))
      x, y = xs[0], ys[0]
      s.set_cursor((y,x))
      global newflood
      newflood = True
      flood_fill_indexes(x, y, (x,y))
  
  s.draw_image()
  s.save(outprefix)

BLACK_TREHSOLD = 250

def to720x720bw(frame):
        try:
            #frame = frame[:,:,3] + (frame[:,:,3]<100)
            frame = 255-frame[:,:,3]
        except:
            frame = frame[:,:,0]
        w,h=frame.shape
        bw = ((frame>BLACK_TREHSOLD)*255 - (frame<=BLACK_TREHSOLD)*frame).astype(np.uint8)
        bw = cv2.resize(bw, (int((h//30)*20), int(w//30*20)), cv2.INTER_LANCZOS4)
        bw = ((bw>254)*255).astype(np.uint8)
        #bw = ((frame>32)*255).astype(np.uint8)
        (miny, minx), (maxy, maxx) = bounding_box_of_black_pixels(bw[:,:])
#bb = bounding_box_of_black_pixels(bw[:,:,1])
        #print(bb)
        w, h = (maxx-minx, maxy-miny)
        if not (w<700 and h < 700):
            print('WARNING: image too large to fit into the inference dimensions')
        print(bw.shape)
        bw = bw[:, minx-10: minx-10+720]
        print(bw.shape)
        return bw

src, dst = sys.argv[1:]
vectorize(src, dst)
