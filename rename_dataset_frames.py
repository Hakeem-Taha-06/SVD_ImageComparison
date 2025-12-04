import os

dataset_dir = r"E:\Ziad\LinearAlgebra_research\ExperimentalCode\data\dataset3_evening_university_walk"

frame_index = 0
for frame in sorted(os.listdir(dataset_dir)):
    os.rename(os.path.join(dataset_dir, frame), os.path.join(dataset_dir, f"frame_{frame_index}.jpg"))
    frame_index += 1    

with open(r"data/labels/dataset3_evening_university_walk.csv", 'w') as f:
    f.write("frame_index,isNovelTruth\n")
    for i in range(289):
        is_novel = 0
        f.write(f"{i},{is_novel}\n")
