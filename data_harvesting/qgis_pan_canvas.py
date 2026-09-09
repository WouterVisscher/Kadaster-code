"""Pan the QGIS map canvas over a 25x25 grid to trigger WMS image requests.

This is a script for the QGIS Python console, not part of the Python package.
It moves the canvas in 100 m steps (EPSG:28992) so the WMS layer issues one
image request per position; those requests are what the request log records.

How to use:
1. Add a WMS layer pointing at the local mapserver (see ``run_wms.sh``).
2. Open View > Debugger/Developer > Requests and press record.
3. Run this script in the QGIS Python console and wait for it to finish.
4. Stop recording and download the request log as JSON (see README.md).
"""

import time

# Current extent of the GUI
canvas = iface.mapCanvas()

# Starting coordinates (top-left corner) in EPSG:28992 (Amersfoort)
x = 210000
y = 462000

# Starting point
point = QgsPointXY(x, y)
canvas.setCenter(point)
canvas.zoomScale(2500)
canvas.refresh()

# Loop over the area
for i in range(25):
    newx = x + i * 100
    for j in range(25):
        loop = QEventLoop()
        newy = y - j * 100
        point = QgsPointXY(newx, newy)
        canvas.setCenter(point)
        canvas.mapCanvasRefreshed.connect(loop.quit)
        canvas.refresh()
        loop.exec_()
        time.sleep(0.1)
