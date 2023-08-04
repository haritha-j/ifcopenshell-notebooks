""" 
helper script to clean up IFC file extracted from Navisworks, by removing 
   a. elements with a certain name
   b. elements which belong to a branch which has other elements with a certain name
"""

import ifcopenshell
from ifcopenshell.util.selector import Selector

from tqdm import tqdm

data_path = "/home/haritha/documents/industrial-facility-relationships/output/temp/"
pipe_name = data_path + "TUBE.ifc"
ip_name = data_path + "TEE.ifc"
op_name = data_path + "ELBOW_ref.ifc"

# remove instances with name FTUBE
def remove_ftubes(ip_name, op_name):
    f1 = open(ip_name, 'r')

    lines = f1.readlines()
    refined_lines = []

    for l in tqdm(lines):
        if 'FTUBE' not in l:
            refined_lines.append(l)
        
    f1.close()

    f2 = open(op_name, 'w')
    f2.writelines(refined_lines)
    f2.close()

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

# remove elbow instances which belong to branches with 'FTUBE's in them
def remove_felbows(ip_name, op_name, pipe_name):
    fpipe = ifcopenshell.open(pipe_name)
    project = fpipe.by_type("IfcProject")
    print ( project)
    selector = Selector()
    fpipe_branches = []

    elements = selector.parse(fpipe, '.IfcPipeSegment[Name *= "FTUBE"]')
    for el in elements:
        branch =  el.Name[el.Name.index('BRANCH')+7:]
        fpipe_branches.append(branch)

    print(fpipe_branches)
    # f1 = ifcopenshell.open(ip_name)
    # selector = Selector()

    # elements = selector.parse(f1, '.IfcPipeFitting')
    # print(len(elements))
    # match_count = 0
    # g = ifcopenshell.file(schema='IFC4')
    # g.add(f1.by_type("IfcProject")[0])

    # for el in tqdm(elements):
    #     name = el.Name
    #     found = False
    #     for branch in fpipe_branches:
    #         if branch in name:
    #             found = True
    #             break
    #     if not found:
    #         g.add(el)

    # g.write(op_name)

    # print(match_count)
    f1 = open(ip_name, 'r')

    lines = f1.readlines()
    refined_lines = []  

    for l in tqdm(lines):
        if 'ELBOW' in l:
            found = False
            for branch in fpipe_branches:
                if branch in l:
                    found = True
                    break
            if not found:
                refined_lines.append(l)
        else:
            refined_lines.append(l)


    f1.close()
    print(len(lines), len(refined_lines))
    f2 = open(op_name, 'w')
    f2.writelines(refined_lines)
    f2.close()


# def remove_felbows(ip_name, op_name, pipe_name):
#     fpipe = open(pipe_name, 'r')
#     f1 = open(ip_name, 'r')

#     pipe_lines = fpipe.readlines()
#     fpipe_branches = []

#     for l in pipe_lines[:1000]:
#         if 'FTUBE' in l:
#             string_occurance = l.index('FTUBE')
#             string_chars = 
        


remove_felbows(ip_name, op_name, pipe_name)
#remove_ftubes(pipe_name, op_name)