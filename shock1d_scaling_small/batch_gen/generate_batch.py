#!/usr/bin/env python3

import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def get_ranks_for_node_count(nodes):
    """Determine ranks to use per node count."""
    if nodes == 1:
        return [1, 2, 4]
    else:
        return [nodes * 4]  # Full node fill with 4 PUs per node

def main():
    parser = argparse.ArgumentParser(description="Generate batch scripts for scalability runs.")
    parser.add_argument("--nodes", type=int, nargs="+", required=True,
                        help="List of node counts (e.g. 1 2 4 8 16)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write batch scripts to")
    parser.add_argument("--emirge-path", type=str, required=True,
                        help="Path to emirge repo (e.g., ../emirge)")
    parser.add_argument("--template", type=Path, default=Path("template_flux.sh.j2"),
                        help="Path to the Jinja2 template")
    parser.add_argument("--input-file", type=str, help="Name of input file")
    parser.add_argument("--casename", type=str, default="shock1d", help="case name")
    parser.add_argument("--bank", type=str, default="uiuc", help="Bank to use")
    parser.add_argument("--time", type=int, default=120,
                        help="Time limit in minutes (default: 120)")
    parser.add_argument("--platform", type=str, default="tuolumne",
                        help="Platform name (for future use)")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    platform=args.platform

    env = Environment(
        loader=FileSystemLoader(args.template.parent),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(args.template.name)

    for nodes in args.nodes:
        ranks = get_ranks_for_node_count(nodes)
        script_content = template.render(
            nodes=nodes,
            time=args.time,
            outfile=f"scal{nodes}.txt",
            ranks=ranks,
            emirge_path=args.emirge_path,
            input_file=args.input_file,
            casename=args.casename,
            bank=args.bank
        )

        script_name = f"run_{nodes}node_{platform}.sh"
        script_path = args.output_dir / script_name
        with open(script_path, "w") as f:
            f.write(script_content)
        script_path.chmod(0o755)
        print(f"Wrote {script_path}")

if __name__ == "__main__":
    main()
