.. Auto Blur documentation master file, created by
   sphinx-quickstart on Sun Sep 15 11:06:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Selective Blur documentation
=======================
Selective Blur is meant as a tool that allows to change the focus of an image on a subject 
selected by the user and adapt the blur of the surroundings naturally. To achieve this it 
leverages `Segment Anything <https://ai.meta.com/sam2>`_ to mask the selected subject and 
`MiDaS <https://github.com/isl-org/MiDaS>`_ for depth-aware blurring.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   modules

