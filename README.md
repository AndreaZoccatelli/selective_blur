# Selective Blur

Selective Blur is a tool that allows to change the focus of an image on a subject 
selected by the user and adapt the blur of the surroundings naturally. 

To achieve this it 
leverages [Segment Anything](https://ai.meta.com/sam2>) to mask the selected subject and 
[MiDaS](https://github.com/isl-org/MiDaS>) for depth-aware blurring.