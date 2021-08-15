def fetch_patches(img, cell_width, cell_height):
  if len(img.shape) == 2:
    height, width = img.shape
  else:
    height, width, channels = img.shape

  if width % cell_width != 0:
    raise Exception("Invalid cell_width %d for width %d" % (cell_width, width))

  if height % cell_height != 0:
    raise Exception("Invalid cell_height %d for height %d" % (cell_height, height))

  cells = []

  for x in range(0, width, cell_width):
    for y in range(0, height, cell_height):
      roi = img[y:y+cell_height, x:x+cell_width]

      cells.append(roi)

  return cells
