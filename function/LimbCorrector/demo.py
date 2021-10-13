from ObjCorrector import ObjCorrector
import os

source_root = "./objs_norm"
target_root = "./objs_norm_corr"

corrector = ObjCorrector()
for filename in os.listdir(source_root):
    source = os.path.join(source_root,filename)
    target = os.path.join(target_root,filename)
    print("processing:",source)
    corrector.run(source,target)

