"""Pan the QGIS canvas over a grid, to record the WMS requests.

Run this file in the QGIS Python console (not outside QGIS): it relies on
the console globals ``iface``, ``QgsPointXY`` and ``QEventLoop``. With the
WMS layer visible and the developer tools recording
(View > Tools > Developer > Requests), the panning generates the HTTP
requests that are saved to a JSON log for
:mod:`labelnet.collect.download_images`.

Note that QGIS caps the number of requests stored in the log, so for
larger areas this has to be repeated several times.
"""

import time

# Current extent of the GUI.
canvas = iface.mapCanvas()

# Start coordinates (top-left corner) in EPSG:28992 (Amersfoort).
x = 210000
y = 462000

point = QgsPointXY(x, y)
canvas.setCenter(point)
canvas.zoomScale(2500)
canvas.refresh()

# Loop over the area.
for i in range(25):
    newx = x + (i + i) * 100
    for j in range(25):
        loop = QEventLoop()
        newy = y - (j + 1) * 100
        point = QgsPointXY(newx, newy)
        canvas.setCenter(point)
        canvas.mapCanvasRefreshed.connect(loop.quit)
        canvas.refresh()
        loop.exec_()
        time.sleep(0.1)
