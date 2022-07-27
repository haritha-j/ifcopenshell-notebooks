import ifcopenshell

new_file = ifcopenshell.file(schema='IFC4')


old_files = list(map(ifcopenshell.open, [
    "deckboxtube_branch.ifc",
        "merged2.ifc"
]))

error_count = 0
for old in old_files:
    for inst in old:
        try:
            new_file.add(inst)
        except:
            error_count += 1

print(error_count)

projects = new_file.by_type("IfcProject")
# Process all but the first project
for p in projects[1:]:
    # Get instances that relate to this project
    refs = new_file.get_inverse(p)
    is_project = lambda inst: inst == p
    assign_first = lambda inst: projects[0]
    # Assign the first project wherever the other project is assigned
    for ref in refs:
        for index, attr in enumerate(ref):
            ref[index] = ifcopenshell.entity_instance.walk(
                is_project,
                assign_first,
                ref[index])
    # Remove from file
    new_file.remove(p)

new_file.write("merged.ifc")