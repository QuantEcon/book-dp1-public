import os
import shutil

julia_src_dir = './source_code_jl/'
python_src_dir = './source_code_py/'
julia_dst_dir = './code-book/jl/'
python_dst_dir = './code-book/py/'

def remove_files(root_src_dir, root_dst_dir):
    for file in os.listdir(root_src_dir):
        if os.path.isfile(os.path.join(root_src_dir, file)):
            os.remove(os.path.join(root_dst_dir, file))
        else:
            shutil.rmtree(os.path.join(root_dst_dir, file))

shutil.copytree(julia_src_dir, julia_dst_dir, dirs_exist_ok=True)
shutil.copytree(python_src_dir, python_dst_dir, dirs_exist_ok=True)

try:
    os.system("jupyter-book build ./code-book/")
    remove_files(python_src_dir, python_dst_dir)
    remove_files(julia_src_dir, julia_dst_dir)
    print("Success!")
    exit(0)
except Exception as e:
    print("Error in building Jupyter Book: " + str(e))
    remove_files(python_src_dir, python_dst_dir)
    remove_files(julia_src_dir, julia_dst_dir)
    print("Failed! - Directory cleaned")
    exit(1)
