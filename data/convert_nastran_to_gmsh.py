import numpy as np
import time

def read_nastran_mesh(file_path):
    nodes = {}
    elements = {}
    element_tags = {}
    physical_names = {}
    physical_names_dim = {}

    with open(file_path, "r") as file:
        lines = file.readlines()

        # Flag variables to determine whether to read nodes or elements
        reading_nodes = False
        reading_elements = False
        reading_physical_names = False

        num_nodes = 0
        num_elements = 0
        num_physical_names = 0
        physical_name = ""

        records = []
        current_record = ""
        # first check for continuation lines, these need to be combined
        for line in lines:
            line = line.rstrip('\n')
            if line.startswith('+'):
                # Lines that start with plus belong to the previous line,
                # remove the + and concatenate
                current_record = current_record.rstrip('+')
                # remove the first 8 character record, this includes the +
                current_record += line[8:]
            else:
                if current_record:
                    records.append(current_record)
                current_record = line

        if current_record:
            records.append(current_record)

        # operate on the concatenated file
        for line in records:
            if line.startswith("$ Node cards"):
                reading_nodes = True
                reading_elements = False
                reading_physical_names = False
                continue
            elif line.startswith("$ Element cards"):
                reading_elements = True
                reading_nodes = False
                reading_physical_names = False
                continue
            elif line.startswith("$ Property cards"):
                reading_elements = False
                reading_nodes = False
                reading_physical_names = True
                continue
            elif line.startswith("$ Material cards"):
                reading_elements = False
                reading_nodes = False
                reading_physical_names = False
                continue

            def read_line(line, field_length=8):
                fields = [line[i:i+field_length] for i in range(0,len(line), field_length)]
                return fields

            if reading_nodes:
                if line.startswith("GRID"):
                    parts = read_line(line)
                    num_fields = len(line)/8
                    node_id = int(parts[1])
                    scale = 1.
                    scale = 0.001
                    x = float(parts[3])*scale
                    y = float(parts[4])*scale
                    z = float(parts[5])*scale if len(parts) > 5 else 0.0
                    nodes[node_id] = (x, y, z)
            elif reading_elements:
                if line.startswith("CQUAD") or\
                   line.startswith("CHEX") or\
                   line.startswith("CROD") or\
                   line.startswith("CTRIA3"):
                    parts = read_line(line)
                    element_id = int(parts[1])
                    element_nodes = [int(parts[i]) for i in range(3, len(parts))]
                    elements[element_id] = element_nodes
                    if element_id in element_tags:
                        element_tags[element_id].append(int(parts[2]))
                    else:
                        element_tags[element_id] = [int(parts[2])]
            elif reading_physical_names:
                if line.startswith("$ Name:"):
                    num_physical_names += 1
                    parts = line.split()
                    physical_name = parts[2]
                    #physical_names[num_physical_names] = physical_name

                if line.startswith("PROD"):
                    pdim = 1
                    #parts = line.split()
                    parts = read_line(line)
                    physical_id = int(parts[1])
                    physical_names_dim[physical_id] = pdim
                    physical_names[physical_id] = physical_name
                elif line.startswith("PSHELL"):
                    pdim = 2
                    #parts = line.split()
                    parts = read_line(line)
                    physical_id = int(parts[1])
                    physical_names_dim[physical_id] = pdim
                    physical_names[physical_id] = physical_name
                elif line.startswith("PSOLID"):
                    pdim = 3
                    #parts = line.split()
                    parts = read_line(line)
                    physical_id = int(parts[1])
                    physical_names_dim[physical_id] = pdim
                    physical_names[physical_id] = physical_name

    # sort the nodes 
    sorted_nodes = dict(sorted(nodes.items()))
    sorted_elements = dict(sorted(elements.items()))


    return sorted_nodes, sorted_elements, element_tags, physical_names, physical_names_dim


def write_gmsh_mesh(file_path, nodes, elements, mesh_format,
                    physical_names, element_tags, physical_dim):
    with open(file_path, 'w') as file:
        # Write mesh format
        file.write("$MeshFormat\n")
        file.write(f"{mesh_format}\n")
        file.write("$EndMeshFormat\n")

        # Write physical names
        file.write("$PhysicalNames\n")
        file.write(f"{len(physical_names)}\n")
        for phys_id, phys_name in physical_names.items():
        #for i, (type_id, number, name) in enumerate(physical_names):
            #file.write(f"{type_id} {number} {name}\n")
            phys_dim = physical_dim[phys_id]
            file.write(f"{phys_dim} {phys_id} \"{phys_name}\"\n")
        file.write("$EndPhysicalNames\n")

        # Write nodes
        file.write("$Nodes\n")
        file.write(f"{len(nodes.items())}\n")
        for node_id, coordinates in nodes.items():
            file.write(f"{node_id} {coordinates[0]} {coordinates[1]} {coordinates[2]}\n")
        file.write("$EndNodes\n")

        # Write elements
        file.write("$Elements\n")
        file.write(f"{len(elements)}\n")
        for elem_id, element_nodes in elements.items():
        #for i, (element_nodes, element_tag) in enumerate(zip(elements, element_tags)):
            if len(element_nodes) == 2:
                   element_type = 1
            elif len(element_nodes) == 3:
                   element_type = 2
            elif len(element_nodes) == 4:
                   element_type = 3
            elif len(element_nodes) == 8:
                   element_type = 5
            else:
                print(f"Unknown element type {element_nodes=}")
                sys.exit(1)
            #element_type = 3 if len(element_nodes) == 4 else 5  # Determine element type based on number of nodes
            tags = element_tags[elem_id]
            num_tags = len(tags)
            file.write(f"{elem_id} {element_type} {num_tags} {' '.join(map(str, tags))} {' '.join(map(str, element_nodes))}\n")
        file.write("$EndElements\n")

def convert_mesh(input_mesh, output_mesh):
    # Read the nastran file
    print(f"Begin reading mesh.")
    begin_read_time = time.perf_counter()
    nodes, elements, element_tags, physical_names, physical_dim = read_nastran_mesh(input_mesh)
    end_read_time = time.perf_counter()
    print(f"Read mesh in {(end_read_time - begin_read_time):.1f} seconds.")

    # This is the mesh format meshmode expects
    mesh_format = "2.2 0 8"

    begin_minmax_time = time.perf_counter()
    first_node = 1e9
    last_node = -1
    for node_id, coordinates in nodes.items():
        first_node = min(first_node, node_id)
        last_node = max(last_node, node_id)
        #print(f"Node {node_id}: {coordinates}")
    print(f"Read {len(nodes)} nodes, first node id {first_node}, last node id {last_node}")

    first_element = 1e9
    last_element = -1
    for element_id, element_nodes in elements.items():
        first_element = min(first_element, element_id)
        last_element = max(last_element, element_id)
        #print(f"Element Nodes for Element {element_id}: {element_nodes}")
    print(f"Read {len(elements)} elements, first element id {first_element}, last element id {last_element}")
    end_minmax_time = time.perf_counter()
    print(f"Spent {(end_minmax_time - begin_minmax_time):.1f} seconds finding min/max node and element numbers.")

    # renumber nodes to start at 1
    begin_renumber_nodes_time = time.perf_counter()
    renumber_nodes = {}
    for new_id, old_id in enumerate(nodes.keys(), start=1):
        renumber_nodes[new_id] = nodes[old_id]

    # update the element node numbers
    renumber_elements = {}
    #renumber_elements = elements
    element_mapping = {}

    # Create a mapping for old node IDs to new IDs
    old_to_new_node_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes.keys(), start=1)}

    # Renumbering nodes
    for new_id, (old_id, coords) in enumerate(nodes.items(), start=1):
        renumber_nodes[new_id] = coords  # Copy coordinates directly

    # Renumbering elements
    for new_id, (element_id, node_ids) in enumerate(list(elements.items()), start=1):
        renumbered_node_ids = [old_to_new_node_mapping[node_id] for node_id in node_ids if node_id in old_to_new_node_mapping]
        renumber_elements[new_id] = renumbered_node_ids  # Store the new node IDs
        element_mapping[element_id] = new_id  # Map old element ID to new ID

    renumber_element_tags = {new_id: element_tags[element_id] for element_id, new_id in element_mapping.items()}

    #renumber_element_tags = element_tags
    end_renumber_nodes_time = time.perf_counter()

    print(f"Spent {(end_renumber_nodes_time - begin_renumber_nodes_time):.1f} seconds renumbering nodes.")

    print("Renumber Nodes:")
    first_node = 1e9
    last_node = -1
    for node_id, coordinates in renumber_nodes.items():
        first_node = min(first_node, node_id)
        last_node = max(last_node, node_id)
        #print(f"Node {node_id}: {coordinates}")
    print(f"Renumbered {len(nodes)} nodes, first node id {first_node}, last node id {last_node}")

    first_element = 1e9
    last_element = -1
    for element_id, element_nodes in renumber_elements.items():
        first_element = min(first_element, element_id)
        last_element = max(last_element, element_id)
        #print(f"Element Nodes for Element {element_id}: {element_nodes}")
    print(f"Renumbered {len(renumber_elements)} elements, first element id {first_element}, last element id {last_element}")

    print("\nPhysical Names:")
    for name_id, name in physical_names.items():
        print(f"Physical name {name_id}: {name}")

    begin_write_time = time.perf_counter()
    write_gmsh_mesh(output_mesh, renumber_nodes, renumber_elements, mesh_format, physical_names, renumber_element_tags, physical_dim)
    end_write_time = time.perf_counter()
    print(f"Wrote new mesh in {(end_write_time - begin_write_time):.1f} seconds.")

import sys
def main():
    if len(sys.argv) != 3:
        print("Usage: convert_nastran_to_gmsh.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    total_time_begin = time.perf_counter()
    convert_mesh(input_mesh=input_file, output_mesh=output_file)
    total_time_end = time.perf_counter()

    print(f"Done converting mesh in {(total_time_end - total_time_begin):.1f} seconds.")

if __name__ == "__main__":
    main()
