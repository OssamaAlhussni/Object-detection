from roboflow import Roboflow

rf = Roboflow(api_key="TusBugZ5GDzBX0CjNWz0")
ws = rf.workspace()
print(ws)
project = rf.workspace("ossamas-workspace-rckyc").project("object-detection-ojs42")
dataset = project.version(1).download("yolov8")