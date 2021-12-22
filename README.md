# position_based_fluids
taichi animated fluid based on "position based fluids" by Macklin M. &amp; Muller M.

Video demo: https://youtu.be/aNsS34amgTU

Before running, need to have taichi in the environment.
run: $pip3 install taichi

Command lines to run:
- stable 2D: $python code_1page.py
- Pusher 2d: $python PBF.py
- Pusher 3d: $python PBF3D.py

The 2D simulation can be viewed in real time.

If you are running in 3d, the above command will generate a serquence of PLY files (100 frames) under /sample.
To view the results, use Houdini, go to Files -> Import Geometry and play!

reference
Macklin M. and Muller M. "Position based fluids", ACM TOG, vol.32, no.4, pp.1-12, 2013
